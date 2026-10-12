//! Deep models on aligned waveform windows (burn). One branch per
//! phenomenology: a small CNN encodes each trace's window (1-D on the
//! waveform, or 2-D on its log-spectrogram), and the traces of an event are
//! pooled (mean and max) so any number of sensors fits. Branches are fused:
//!
//! - **late**: each branch has its own classifier; a learned gate averages
//!   the present branches' probabilities. Training adds each branch's own
//!   loss, so every branch classifies alone.
//! - **early**: pooled branch embeddings (zero when absent) and presence
//!   flags feed one classifier.
//!
//! Training randomly drops present branches (`branch_dropout`) so the fused
//! model copes with a missing phenomenology. Inference runs on the pure-Rust
//! CPU device; weights are stored as safetensors.

use burn::module::{Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig1d, PaddingConfig2d};
use burn::optim::{AdamConfig, GradientsParams};
use burn::store::{ModuleSnapshot, SafetensorsStore};
use burn::tensor::{Device, Int, Tensor, TensorData, activation};
use serde::{Deserialize, Serialize};
use squiggle_core::transform::{Complex, fft};
use squiggle_core::windows::{WindowSpec, Windows};

use crate::classical::Params;
use crate::rng::Rng;

/// Spectrogram frame and hop, samples.
const FRAME: usize = 32;
const HOP: usize = 16;
/// Additive mask for absent entries.
const ABSENT: f32 = -1e4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Encoder {
    /// 1-D CNN on the window.
    Waveform,
    /// 2-D CNN on the window's log-spectrogram.
    Spectrogram,
}

impl Encoder {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "waveform" => Some(Self::Waveform),
            "spectrogram" => Some(Self::Spectrogram),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Fusion {
    Late,
    Early,
}

impl Fusion {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "late" => Some(Self::Late),
            "early" => Some(Self::Early),
            _ => None,
        }
    }
}

/// One phenomenology's input.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Branch {
    /// A prediction needs at least one window from every required branch.
    pub required: bool,
    pub encoder: Encoder,
    pub window: WindowSpec,
}

/// Everything needed to rebuild the network before loading its weights.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Architecture {
    pub fusion: Fusion,
    pub classes: usize,
    /// Channels of each encoder's last layer; a trace embeds as twice this.
    pub width: usize,
    pub hidden: usize,
    pub branches: Vec<Branch>,
}

/// Log-power spectrogram of a window, `FRAME / 2` bins by frames, row-major, standardized.
pub fn spectrogram(window: &[f32]) -> Vec<f32> {
    let frames = spectrogram_frames(window.len());
    let bins = FRAME / 2;
    let hann: Vec<f64> = (0..FRAME).map(|i| 0.5 - 0.5 * (core::f64::consts::TAU * i as f64 / FRAME as f64).cos()).collect();
    let mut out = vec![0.0f64; bins * frames];
    for f in 0..frames {
        let mut data: Vec<Complex> = (0..FRAME).map(|i| Complex::new(f64::from(window[f * HOP + i]) * hann[i], 0.0)).collect();
        fft(&mut data);
        for b in 0..bins {
            let c = data[b + 1];
            out[b * frames + f] = (c.re * c.re + c.im * c.im + 1e-6).log10();
        }
    }
    let n = out.len().max(1) as f64;
    let mean = out.iter().sum::<f64>() / n;
    let sd = (out.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt().max(1e-6);
    out.into_iter().map(|v| ((v - mean) / sd) as f32).collect()
}

fn spectrogram_frames(samples: usize) -> usize {
    if samples < FRAME { 0 } else { (samples - FRAME) / HOP + 1 }
}

#[derive(Module, Debug)]
pub struct WaveEncoder {
    c1: Conv1d,
    c2: Conv1d,
    c3: Conv1d,
}

impl WaveEncoder {
    fn new(width: usize, device: &Device) -> Self {
        let conv = |i, o, k| Conv1dConfig::new(i, o, k).with_stride(2).with_padding(PaddingConfig1d::Same).init(device);
        Self { c1: conv(1, 8, 7), c2: conv(8, 16, 5), c3: conv(16, width, 5) }
    }

    /// `[traces, 1, samples]` → `[traces, 2 · width]` (mean and max over time).
    fn forward(&self, x: Tensor<3>) -> Tensor<2> {
        let h = activation::relu(self.c1.forward(x));
        let h = activation::relu(self.c2.forward(h));
        let h = activation::relu(self.c3.forward(h));
        let mean: Tensor<2> = h.clone().mean_dim(2).squeeze_dim(2);
        let max: Tensor<2> = h.max_dim(2).squeeze_dim(2);
        Tensor::cat(vec![mean, max], 1)
    }
}

#[derive(Module, Debug)]
pub struct SpectrogramEncoder {
    c1: Conv2d,
    c2: Conv2d,
    c3: Conv2d,
}

impl SpectrogramEncoder {
    fn new(width: usize, device: &Device) -> Self {
        let conv = |i, o, s| Conv2dConfig::new([i, o], [3, 3]).with_stride([s, s]).with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1)).init(device);
        Self { c1: conv(1, 8, 1), c2: conv(8, 16, 2), c3: conv(16, width, 2) }
    }

    /// `[traces, 1, bins, frames]` → `[traces, 2 · width]`.
    fn forward(&self, x: Tensor<4>) -> Tensor<2> {
        let h = activation::relu(self.c1.forward(x));
        let h = activation::relu(self.c2.forward(h));
        let h = activation::relu(self.c3.forward(h));
        let [n, c, a, b] = h.dims();
        let h: Tensor<3> = h.reshape([n, c, a * b]);
        let mean: Tensor<2> = h.clone().mean_dim(2).squeeze_dim(2);
        let max: Tensor<2> = h.max_dim(2).squeeze_dim(2);
        Tensor::cat(vec![mean, max], 1)
    }
}

#[derive(Module, Debug)]
pub struct Head {
    a: Linear,
    b: Linear,
}

impl Head {
    fn new(input: usize, hidden: usize, classes: usize, device: &Device) -> Self {
        Self { a: LinearConfig::new(input, hidden).init(device), b: LinearConfig::new(hidden, classes).init(device) }
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.b.forward(activation::relu(self.a.forward(x)))
    }
}

#[derive(Module, Debug)]
pub struct BranchNet {
    wave: Option<WaveEncoder>,
    spectrogram: Option<SpectrogramEncoder>,
    /// Late fusion only.
    head: Option<Head>,
}

#[derive(Module, Debug)]
pub struct Net {
    branches: Vec<BranchNet>,
    /// Late fusion: each branch's vote before masking absent ones.
    gate: Option<Param<Tensor<1>>>,
    /// Early fusion.
    joint: Option<Head>,
}

/// One branch of a batch: every trace window, and the event each belongs to.
struct BranchInput {
    windows: Vec<Vec<f32>>,
    owners: Vec<usize>,
}

/// Network outputs for a batch of events.
struct Output {
    /// `[events, classes]`.
    probabilities: Tensor<2>,
    /// Late fusion: per-branch log-probabilities `[events, classes]`.
    branches: Vec<Tensor<2>>,
    /// `[events][branch]`.
    present: Vec<Vec<bool>>,
}

impl Net {
    pub fn new(arch: &Architecture, device: &Device) -> Self {
        let embed = 4 * arch.width;
        let late = arch.fusion == Fusion::Late;
        let branches = arch
            .branches
            .iter()
            .map(|b| BranchNet {
                wave: (b.encoder == Encoder::Waveform).then(|| WaveEncoder::new(arch.width, device)),
                spectrogram: (b.encoder == Encoder::Spectrogram).then(|| SpectrogramEncoder::new(arch.width, device)),
                head: late.then(|| Head::new(embed, arch.hidden, arch.classes, device)),
            })
            .collect();
        let n = arch.branches.len();
        Self {
            branches,
            gate: late.then(|| Param::from_tensor(Tensor::zeros([n], device))),
            joint: (!late).then(|| Head::new(n * (embed + 1), arch.hidden, arch.classes, device)),
        }
    }

    fn forward(&self, arch: &Architecture, inputs: &[BranchInput], events: usize, device: &Device) -> Output {
        let embed = 4 * arch.width;
        let mut present = vec![vec![false; inputs.len()]; events];
        let mut pooled = Vec::with_capacity(inputs.len());
        for (b, (net, input)) in self.branches.iter().zip(inputs).enumerate() {
            let traces = input.windows.len();
            let mut counts = vec![0usize; events];
            for &o in &input.owners {
                counts[o] += 1;
                present[o][b] = true;
            }
            if traces == 0 {
                pooled.push(Tensor::<2>::zeros([events, embed], device));
                continue;
            }
            let samples = input.windows[0].len();
            let encoded = match (&net.wave, &net.spectrogram) {
                (Some(wave), _) => {
                    let flat: Vec<f32> = input.windows.iter().flatten().copied().collect();
                    wave.forward(Tensor::from_data(TensorData::new(flat, [traces, 1, samples]), device))
                }
                (None, Some(spec)) => {
                    let frames = spectrogram_frames(samples);
                    let flat: Vec<f32> = input.windows.iter().flat_map(|w| spectrogram(w)).collect();
                    spec.forward(Tensor::from_data(TensorData::new(flat, [traces, 1, FRAME / 2, frames]), device))
                }
                (None, None) => unreachable!("every branch has an encoder"),
            };
            let d = encoded.dims()[1];
            // Mean over each event's traces, as a matrix product.
            let mut weights = vec![0.0f32; events * traces];
            let mut mask = vec![ABSENT; events * traces];
            for (t, &o) in input.owners.iter().enumerate() {
                weights[o * traces + t] = 1.0 / counts[o] as f32;
                mask[o * traces + t] = 0.0;
            }
            let mean = Tensor::<2>::from_data(TensorData::new(weights, [events, traces]), device).matmul(encoded.clone());
            let mask = Tensor::<3>::from_data(TensorData::new(mask, [events, traces, 1]), device);
            let max: Tensor<2> = (encoded.reshape([1, traces, d]) + mask).max_dim(1).squeeze_dim(1);
            let has: Vec<f32> = counts.iter().map(|&c| if c > 0 { 1.0 } else { 0.0 }).collect();
            let has = Tensor::<2>::from_data(TensorData::new(has, [events, 1]), device);
            pooled.push(Tensor::cat(vec![mean, max * has], 1));
        }

        match (&self.gate, &self.joint) {
            (Some(gate), _) => {
                let logs: Vec<Tensor<2>> = self
                    .branches
                    .iter()
                    .zip(pooled)
                    .map(|(net, x)| activation::log_softmax(net.head.as_ref().expect("late fusion has heads").forward(x), 1))
                    .collect();
                let mask: Vec<f32> = present.iter().flat_map(|row| row.iter().map(|&p| if p { 0.0 } else { ABSENT })).collect();
                let mask = Tensor::<2>::from_data(TensorData::new(mask, [events, inputs.len()]), device);
                let w = activation::softmax(gate.val().reshape([1, inputs.len()]) + mask, 1);
                let mut probabilities = Tensor::<2>::zeros([events, arch.classes], device);
                for (b, log) in logs.iter().enumerate() {
                    let wb: Tensor<2> = w.clone().slice([0..events, b..b + 1]);
                    probabilities = probabilities + log.clone().exp() * wb;
                }
                Output { probabilities, branches: logs, present }
            }
            (None, Some(joint)) => {
                let flags: Vec<Tensor<2>> = (0..inputs.len())
                    .map(|b| {
                        let v: Vec<f32> = present.iter().map(|row| if row[b] { 1.0 } else { 0.0 }).collect();
                        Tensor::from_data(TensorData::new(v, [events, 1]), device)
                    })
                    .collect();
                let mut parts = pooled;
                parts.extend(flags);
                let probabilities = activation::softmax(joint.forward(Tensor::cat(parts, 1)), 1);
                Output { probabilities, branches: Vec::new(), present }
            }
            (None, None) => unreachable!("one fusion is configured"),
        }
    }
}

/// Training device: the pure-Rust CPU, or with the `gpu` feature the default wgpu adapter.
pub fn device(gpu: bool) -> Result<Device, String> {
    if !gpu {
        return Ok(Device::flex());
    }
    #[cfg(feature = "gpu")]
    return Ok(Device::wgpu(burn::tensor::DeviceKind::DefaultDevice));
    #[cfg(not(feature = "gpu"))]
    Err("GPU training needs squiggle-train built with --features gpu".to_string())
}

/// One event's training input: windows per branch (in architecture order) and its class.
#[derive(Clone, Debug)]
pub struct Example {
    pub windows: Vec<Vec<Vec<f32>>>,
    pub class: usize,
}

/// A trained network on the inference device.
#[derive(Debug)]
pub struct DeepModel {
    pub arch: Architecture,
    net: Net,
    device: Device,
}

/// Probabilities, and per present branch (late fusion) its own.
pub type DeepOutput = (Vec<f64>, Vec<(usize, Vec<f64>)>);

impl DeepModel {
    /// Fits on `device` (autodiff is enabled here); the result runs on the CPU.
    pub fn train(arch: &Architecture, examples: &[Example], params: &Params, seed: u64, device: &Device) -> Self {
        let device = device.clone().autodiff();
        device.seed(seed);
        let mut rng = Rng::new(seed);
        let mut net = Net::new(arch, &device);
        let mut optimizer = AdamConfig::new().init();
        let mut order: Vec<usize> = (0..examples.len()).collect();
        for _ in 0..params.epochs {
            rng.shuffle(&mut order);
            for batch in order.chunks(params.batch.max(1)) {
                let mut inputs: Vec<BranchInput> = arch.branches.iter().map(|_| BranchInput { windows: Vec::new(), owners: Vec::new() }).collect();
                for (e, &i) in batch.iter().enumerate() {
                    let example = &examples[i];
                    let present: Vec<usize> = (0..arch.branches.len()).filter(|&b| !example.windows[b].is_empty()).collect();
                    // Drop present branches at random, keeping at least one.
                    let keep = rng.below(present.len().max(1));
                    for (k, &b) in present.iter().enumerate() {
                        if k != keep && present.len() > 1 && rng.uniform() < params.branch_dropout {
                            continue;
                        }
                        for w in &example.windows[b] {
                            inputs[b].windows.push(w.clone());
                            inputs[b].owners.push(e);
                        }
                    }
                }
                let classes: Vec<i64> = batch.iter().map(|&i| examples[i].class as i64).collect();
                let target = Tensor::<1, Int>::from_data(TensorData::new(classes, [batch.len()]), &device).reshape([batch.len(), 1]);
                let out = net.forward(arch, &inputs, batch.len(), &device);
                let picked: Tensor<2> = (out.probabilities + 1e-7).log().gather(1, target.clone());
                let mut loss: Tensor<1> = picked.mean().neg();
                // Late fusion: each present branch also classifies alone.
                for (b, log) in out.branches.iter().enumerate() {
                    let mask: Vec<f32> = out.present.iter().map(|row| if row[b] { 1.0 } else { 0.0 }).collect();
                    let count: f32 = mask.iter().sum();
                    if count > 0.0 {
                        let mask = Tensor::<2>::from_data(TensorData::new(mask, [batch.len(), 1]), &device);
                        let nll: Tensor<1> = (log.clone().gather(1, target.clone()) * mask).sum().neg() / count;
                        loss = loss + nll;
                    }
                }
                let grads = GradientsParams::from_grads(loss.backward(), &net);
                net = optimizer.step(params.learning_rate, net, grads);
            }
        }
        let cpu = Device::flex();
        Self { arch: arch.clone(), net: net.valid().to_device(&cpu), device: cpu }
    }

    /// Weights as safetensors.
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let mut store = SafetensorsStore::from_bytes(None);
        self.net.save_into(&mut store).map_err(|e| e.to_string())?;
        store.get_bytes().map_err(|e| e.to_string())
    }

    pub fn from_bytes(arch: &Architecture, bytes: &[u8]) -> Result<Self, String> {
        let device = Device::flex();
        let mut net = Net::new(arch, &device);
        let mut store = SafetensorsStore::from_bytes(Some(bytes.to_vec()));
        net.load_from(&mut store).map_err(|e| e.to_string())?;
        Ok(Self { arch: arch.clone(), net, device })
    }

    /// `None` when a required branch, or every branch, has no window.
    pub fn predict(&self, windows: &[Windows]) -> Option<DeepOutput> {
        let present: Vec<bool> = windows.iter().map(|w| !w.traces.is_empty()).collect();
        if !present.iter().any(|p| *p) || self.arch.branches.iter().zip(&present).any(|(b, p)| b.required && !p) {
            return None;
        }
        let inputs: Vec<BranchInput> = windows.iter().map(|w| BranchInput { windows: w.traces.clone(), owners: vec![0; w.traces.len()] }).collect();
        let out = self.net.forward(&self.arch, &inputs, 1, &self.device);
        let row = |t: Tensor<2>| -> Vec<f64> { t.into_data().iter::<f32>().map(f64::from).collect() };
        let parts = out.branches.into_iter().enumerate().filter(|(b, _)| present[*b]).map(|(b, log)| (b, row(log.exp()))).collect();
        Some((row(out.probabilities), parts))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use squiggle_core::PhenomenologyId;

    fn spec(phenomenology: &str) -> WindowSpec {
        WindowSpec { phenomenology: PhenomenologyId::new(phenomenology), rate: 40.0, pre_s: 0.5, samples: 128, max_traces: 4 }
    }

    /// Class 0: low tone on "a", nothing on "b". Class 1: high tone on "a", a pulse on "b".
    fn example(rng: &mut Rng, class: usize) -> Example {
        let tone = |f: f64, rng: &mut Rng| -> Vec<f32> {
            let phase = rng.uniform() * 6.0;
            (0..128).map(|i| ((core::f64::consts::TAU * f * i as f64 / 40.0 + phase).sin() + 0.2 * rng.normal()) as f32).collect()
        };
        let a = (0..2).map(|_| tone(if class == 0 { 2.0 } else { 9.0 }, rng)).collect();
        let b = vec![(0..128).map(|i| if class == 1 && (20..30).contains(&i) { 1.0 } else { 0.1 * rng.normal() as f32 }).collect()];
        Example { windows: vec![a, b], class }
    }

    fn arch(fusion: Fusion, encoder: Encoder) -> Architecture {
        Architecture {
            fusion,
            classes: 2,
            width: 8,
            hidden: 16,
            branches: vec![Branch { required: false, encoder, window: spec("a") }, Branch { required: false, encoder: Encoder::Waveform, window: spec("b") }],
        }
    }

    fn windows(example: &Example, keep: [bool; 2]) -> Vec<Windows> {
        example.windows.iter().zip(keep).map(|(w, k)| Windows { traces: if k { w.clone() } else { vec![] }, sources: vec![] }).collect()
    }

    #[test]
    fn fused_networks_classify_with_a_branch_missing() {
        let mut rng = Rng::new(3);
        let train: Vec<Example> = (0..40).map(|i| example(&mut rng, i % 2)).collect();
        let test: Vec<Example> = (0..10).map(|i| example(&mut rng, i % 2)).collect();
        let params = Params { epochs: 12, batch: 8, learning_rate: 0.01, ..Params::default() };
        for (fusion, encoder) in [(Fusion::Late, Encoder::Waveform), (Fusion::Early, Encoder::Spectrogram)] {
            let arch = arch(fusion, encoder);
            let model = DeepModel::train(&arch, &train, &params, 1, &Device::flex());
            for keep in [[true, true], [true, false], [false, true]] {
                let correct = test.iter().filter(|e| crate::argmax(&model.predict(&windows(e, keep)).unwrap().0) == e.class).count();
                assert!(correct >= 9, "{fusion:?} {encoder:?} {keep:?}: {correct}/10");
            }
            let (p, parts) = model.predict(&windows(&test[1], [true, false])).unwrap();
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-5);
            assert_eq!(parts.iter().map(|p| p.0).collect::<Vec<_>>(), if fusion == Fusion::Late { vec![0] } else { vec![] });
            assert!(model.predict(&windows(&test[1], [false, false])).is_none());

            let loaded = DeepModel::from_bytes(&arch, &model.to_bytes().unwrap()).unwrap();
            assert_eq!(loaded.predict(&windows(&test[0], [true, true])), model.predict(&windows(&test[0], [true, true])));
        }
        let mut required = arch(Fusion::Late, Encoder::Waveform);
        required.branches[1].required = true;
        let model = DeepModel::train(&required, &train[..4], &Params { epochs: 1, ..params }, 1, &Device::flex());
        assert!(model.predict(&windows(&test[0], [true, false])).is_none());
    }

    #[test]
    fn spectrograms_show_a_tone_in_its_bin() {
        // 10 Hz at 40 Hz rate: bin width 40/32 = 1.25 Hz, so bin 8 (index 7 after skipping DC).
        let window: Vec<f32> = (0..128).map(|i| (core::f64::consts::TAU * 10.0 * i as f64 / 40.0).sin() as f32).collect();
        let s = spectrogram(&window);
        let frames = spectrogram_frames(128);
        assert_eq!((s.len(), frames), (16 * 7, 7));
        let loudest = (0..16).max_by(|&a, &b| s[a * frames].total_cmp(&s[b * frames])).unwrap();
        assert_eq!(loudest, 7);
    }
}
