//! `squiggle-train <dataset> --out <package> [options]`, or
//! `squiggle-train cluster <dataset> --out <package> …`, or
//! `squiggle-train onnx <model.onnx> --out <package> …`: see `USAGE`.

use std::path::PathBuf;
use std::process::ExitCode;

use squiggle_models::Kind;
use squiggle_models::deep::{Encoder, Fusion};
use squiggle_train::{ClusterOptions, Dataset, Level, Method, Options, predictions, train};

const USAGE: &str = "\
usage: squiggle-train <dataset-dir> --out <package-dir> [options]
       squiggle-train cluster <dataset-dir> --out <package-dir> [--features <a,b,…>]
                      [--components <n>] [--eps <x>] [--min-points <n>] [--name <name>] [--version <n>]
       squiggle-train onnx <model.onnx> --out <package-dir> --features <a,b,…> --labels <x,y,…>
                      [--name <name>] [--version <n>] [--taxonomy-version <n>]   (onnx feature)

  --model <kind>        logistic | svm | knn | tree | forest on event features,
                        or cnn on waveform windows (default forest)
  --encoder <e>         cnn: waveform (1-D) | spectrogram (2-D) (default waveform)
  --fusion <f>          cnn: late (per-phenomenology heads, gated) | early (joint head) (default late)
  --branches <a,b,…>    cnn: phenomenologies to use (default every one with windows)
  --require <a,b,…>     cnn: phenomenologies a prediction needs (default none)
  --gpu                 cnn: train on the GPU (gpu feature)
  --name <name>         model name (default: the kind)
  --version <n>         model version (default 1)
  --level <level>       leaf (analyst label) | top (root ancestor) (default leaf)
  --no-noise            leave out rejected events instead of training a noise class
  --features <a,b,…>    event features to use (default all)
  --seed <n>            random seed (default 1)
  --folds <n>           cross-validation folds, at most the smallest class (default 5)
  --l2 <x> --iterations <n> --rate <x> --k <n> --max-depth <n> --min-leaf <n> --trees <n>
  --epochs <n> --batch <n> --learning-rate <x> --branch-dropout <x> --width <n> --hidden <n>
                        hyperparameters
  --predictions <file>  also write each event's probabilities (cluster: coordinates and cluster) as CSV

cluster: PCA of the standardized event features of every event, labelled or
not, then DBSCAN in the PCA space. --components 0 keeps 90 % of the variance;
--eps 0 picks the radius from the data; --min-points defaults to 3.";

fn main() -> ExitCode {
    match run(std::env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("squiggle-train: {error}");
            ExitCode::FAILURE
        }
    }
}

fn list(v: String) -> Vec<String> {
    v.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
}

fn number<T: std::str::FromStr>(arg: &str, v: String) -> Result<T, String> {
    v.parse().map_err(|_| format!("{arg}: not a number: {v}"))
}

fn run(args: Vec<String>) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("onnx") => return onnx(args.into_iter().skip(1).collect()),
        Some("cluster") => return cluster(args.into_iter().skip(1).collect()),
        _ => {}
    }
    let mut options = Options::default();
    let (mut encoder, mut fusion, mut cnn) = (Encoder::Waveform, Fusion::Late, false);
    let mut dataset: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut csv: Option<PathBuf> = None;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--out" => out = Some(value()?.into()),
            "--model" => {
                let v = value()?;
                cnn = v == "cnn";
                if !cnn {
                    options.method = Method::Classical(Kind::parse(&v).ok_or_else(|| format!("unknown model {v}"))?);
                }
            }
            "--encoder" => {
                let v = value()?;
                encoder = Encoder::parse(&v).ok_or_else(|| format!("unknown encoder {v}"))?;
            }
            "--fusion" => {
                let v = value()?;
                fusion = Fusion::parse(&v).ok_or_else(|| format!("unknown fusion {v}"))?;
            }
            "--branches" => options.branches = Some(list(value()?)),
            "--require" => options.require = list(value()?),
            "--gpu" => options.gpu = true,
            "--epochs" => options.params.epochs = number(&arg, value()?)?,
            "--batch" => options.params.batch = number(&arg, value()?)?,
            "--learning-rate" => options.params.learning_rate = number(&arg, value()?)?,
            "--branch-dropout" => options.params.branch_dropout = number(&arg, value()?)?,
            "--width" => options.params.width = number(&arg, value()?)?,
            "--hidden" => options.params.hidden = number(&arg, value()?)?,
            "--name" => options.name = Some(value()?),
            "--version" => options.version = number(&arg, value()?)?,
            "--level" => {
                options.level = match value()?.as_str() {
                    "leaf" => Level::Leaf,
                    "top" => Level::Top,
                    v => return Err(format!("unknown level {v}")),
                }
            }
            "--no-noise" => options.noise = false,
            "--features" => options.features = Some(list(value()?)),
            "--seed" => options.seed = number(&arg, value()?)?,
            "--folds" => options.folds = number(&arg, value()?)?,
            "--l2" => options.params.l2 = number(&arg, value()?)?,
            "--iterations" => options.params.iterations = number(&arg, value()?)?,
            "--rate" => options.params.rate = number(&arg, value()?)?,
            "--k" => options.params.k = number(&arg, value()?)?,
            "--max-depth" => options.params.max_depth = number(&arg, value()?)?,
            "--min-leaf" => options.params.min_leaf = number(&arg, value()?)?,
            "--trees" => options.params.trees = number(&arg, value()?)?,
            "--predictions" => csv = Some(value()?.into()),
            _ if arg.starts_with('-') => return Err(format!("unknown option {arg}\n{USAGE}")),
            _ if dataset.is_none() => dataset = Some(arg.into()),
            _ => return Err(format!("unexpected argument {arg}\n{USAGE}")),
        }
    }
    let (Some(dataset), Some(out)) = (dataset, out) else {
        return Err(USAGE.to_string());
    };
    if cnn {
        options.method = Method::Cnn { encoder, fusion };
    }

    let dataset = Dataset::load(&dataset)?;
    let package = train(&dataset, &options)?;
    package.save(&out)?;

    let spec = &package.spec;
    let t = &spec.training;
    println!("{} → {}", spec.id, out.display());
    let classes: Vec<String> = t.classes.iter().map(|c| format!("{} {}", c.events, c.label)).collect();
    println!("trained on {} events: {}", t.events, classes.join(", "));
    println!("fit accuracy {:.3}, log loss {:.3}", t.fit.accuracy, t.fit.log_loss);
    if t.folds > 0 {
        let cv = &t.cross_validation;
        println!("{}-fold accuracy {:.3}, log loss {:.3}; confusion (rows true, columns predicted):", t.folds, cv.accuracy, cv.log_loss);
        let width = spec.output.labels.iter().map(String::len).max().unwrap_or(0);
        for (label, row) in spec.output.labels.iter().zip(&cv.confusion) {
            let cells: Vec<String> = row.iter().map(|n| format!("{n:>5}")).collect();
            println!("  {label:>width$} {}", cells.join(""));
        }
    } else {
        println!("too few events per class to cross-validate");
    }

    if let Some(path) = csv {
        let mut text = format!("event,{}\n", spec.output.labels.join(","));
        for (event, p) in predictions(&package, &dataset) {
            let cells: Vec<String> = match p {
                Some(p) => p.iter().map(|v| v.to_string()).collect(),
                None => vec![String::new(); spec.output.labels.len()],
            };
            text.push_str(&format!("{event},{}\n", cells.join(",")));
        }
        std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
        println!("predictions → {}", path.display());
    }
    Ok(())
}

fn cluster(args: Vec<String>) -> Result<(), String> {
    let mut options = ClusterOptions { version: 1, ..Default::default() };
    let (mut dataset, mut out, mut csv) = (None::<PathBuf>, None::<PathBuf>, None::<PathBuf>);
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--out" => out = Some(value()?.into()),
            "--features" => options.features = Some(list(value()?)),
            "--components" => options.params.components = number(&arg, value()?)?,
            "--eps" => options.params.eps = number(&arg, value()?)?,
            "--min-points" => options.params.min_points = number(&arg, value()?)?,
            "--name" => options.name = Some(value()?),
            "--version" => options.version = number(&arg, value()?)?,
            "--predictions" => csv = Some(value()?.into()),
            _ if arg.starts_with('-') => return Err(format!("unknown option {arg}\n{USAGE}")),
            _ if dataset.is_none() => dataset = Some(arg.into()),
            _ => return Err(format!("unexpected argument {arg}\n{USAGE}")),
        }
    }
    let (Some(dir), Some(out)) = (dataset, out) else {
        return Err(USAGE.to_string());
    };
    let dataset = Dataset::load(&dir)?;
    let (package, assigned) = squiggle_train::cluster(&dataset, &options)?;
    package.save(&out)?;
    let spec = &package.spec;
    let fitted = spec.cluster.unwrap_or_default();
    println!("{} → {}", spec.id, out.display());
    println!("{} events, {} components, eps {:.3}, min points {}", spec.training.events, fitted.components, fitted.eps, fitted.min_points);
    let noise = assigned.iter().filter(|a| a.is_none()).count();
    println!("{} clusters, {noise} unclustered", spec.output.clusters.len());
    for c in &spec.output.clusters {
        let classes: Vec<String> = c.classes.iter().map(|k| format!("{} {}", k.events, k.label)).collect();
        println!("  {:>3}: {:>4} events  {}", c.cluster, c.events, classes.join(", "));
    }
    let truth: Vec<Option<String>> = dataset.classes.clone();
    if truth.iter().any(Option::is_some) {
        let mut names: Vec<&String> = truth.iter().flatten().collect();
        names.sort();
        names.dedup();
        let classes: Vec<Option<usize>> = truth.iter().map(|c| c.as_ref().and_then(|c| names.iter().position(|n| *n == c))).collect();
        println!("adjusted Rand index against the classes: {:.3}", squiggle_train::adjusted_rand(&assigned, &classes));
    }
    if let Some(path) = csv {
        let mut text = String::from("event,cluster");
        for k in 0..fitted.components {
            text.push_str(&format!(",pc{}", k + 1));
        }
        text.push('\n');
        for (i, (event, row)) in dataset.events.iter().zip(&dataset.rows).enumerate() {
            let (point, _) = package.embed(row).expect("a cluster package");
            let cells: Vec<String> = point.iter().map(|v| v.to_string()).collect();
            text.push_str(&format!("{event},{},{}\n", assigned[i].map_or(String::new(), |k| k.to_string()), cells.join(",")));
        }
        std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
        println!("coordinates → {}", path.display());
    }
    Ok(())
}

#[cfg(feature = "onnx")]
fn onnx(args: Vec<String>) -> Result<(), String> {
    let (mut model, mut out, mut features, mut labels) = (None::<PathBuf>, None::<PathBuf>, Vec::new(), Vec::new());
    let (mut name, mut version, mut taxonomy) = ("onnx".to_string(), 1u32, 0u32);
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--out" => out = Some(value()?.into()),
            "--features" => features = list(value()?),
            "--labels" => labels = list(value()?),
            "--name" => name = value()?,
            "--version" => version = number(&arg, value()?)?,
            "--taxonomy-version" => taxonomy = number(&arg, value()?)?,
            _ if arg.starts_with('-') => return Err(format!("unknown option {arg}\n{USAGE}")),
            _ => model = Some(arg.into()),
        }
    }
    let (Some(model), Some(out)) = (model, out) else {
        return Err(USAGE.to_string());
    };
    let bytes = std::fs::read(&model).map_err(|e| format!("{}: {e}", model.display()))?;
    let package = squiggle_train::package_onnx(bytes, features, labels, &name, version, taxonomy)?;
    package.save(&out)?;
    println!("{} → {}", package.spec.id, out.display());
    Ok(())
}

#[cfg(not(feature = "onnx"))]
fn onnx(_: Vec<String>) -> Result<(), String> {
    Err("ONNX packaging needs squiggle-train built with --features onnx".to_string())
}
