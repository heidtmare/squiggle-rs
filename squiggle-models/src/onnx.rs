//! Imported ONNX models on event features, run with tract (pure Rust).
//! The graph takes `[1, n]` f32 features in the spec's column order (missing
//! values as 0) and returns `[1, classes]`; outputs that are not already
//! probabilities go through a softmax.

use tract_onnx::prelude::*;

use crate::classical::softmax;

type Runnable = std::sync::Arc<TypedRunnableModel>;

#[derive(Debug)]
pub struct OnnxModel {
    model: Runnable,
    columns: Vec<usize>,
    classes: usize,
}

impl OnnxModel {
    pub fn load(bytes: &[u8], columns: Vec<usize>, classes: usize) -> Result<Self, String> {
        let model = tract_onnx::onnx()
            .model_for_read(&mut &bytes[..])
            .and_then(|m| m.with_input_fact(0, f32::fact([1, columns.len()]).into()))
            .and_then(|m| m.into_optimized())
            .and_then(|m| m.into_runnable())
            .map_err(|e| format!("model.onnx: {e}"))?;
        Ok(Self { model, columns, classes })
    }

    pub fn row(&self, event: &[f64]) -> Vec<f64> {
        self.columns.iter().map(|&c| event.get(c).copied().unwrap_or(f64::NAN)).collect()
    }

    pub fn predict(&self, event: &[f64]) -> Option<Vec<f64>> {
        let row: Vec<f32> = self.row(event).into_iter().map(|v| if v.is_finite() { v as f32 } else { 0.0 }).collect();
        let input = tract_ndarray::Array2::from_shape_vec((1, row.len()), row).ok()?;
        let outputs = self.model.run(tvec!(Tensor::from(input).into())).ok()?;
        let values: Vec<f64> = outputs.first()?.to_plain_array_view::<f32>().ok()?.iter().map(|v| f64::from(*v)).collect();
        if values.len() != self.classes {
            return None;
        }
        let probabilities = values.iter().all(|v| (0.0..=1.0).contains(v)) && (values.iter().sum::<f64>() - 1.0).abs() < 1e-3;
        Some(if probabilities { values } else { softmax(&values) })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use prost::Message;
    use tract_onnx::pb::{
        GraphProto, ModelProto, NodeProto, OperatorSetIdProto, TensorProto, TensorShapeProto, TypeProto, ValueInfoProto, tensor_shape_proto, type_proto,
    };

    fn value(name: &str, dims: &[i64]) -> ValueInfoProto {
        let dim = dims.iter().map(|d| tensor_shape_proto::Dimension { value: Some(tensor_shape_proto::dimension::Value::DimValue(*d)), ..Default::default() }).collect();
        ValueInfoProto {
            name: name.into(),
            r#type: Some(TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor { elem_type: 1, shape: Some(TensorShapeProto { dim }) })),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn node(op: &str, inputs: &[&str], output: &str) -> NodeProto {
        NodeProto { op_type: op.into(), input: inputs.iter().map(|s| s.to_string()).collect(), output: vec![output.into()], name: output.into(), ..Default::default() }
    }

    /// softmax(x · w + b) for `[1, 2]` features and two classes.
    pub(crate) fn linear_softmax(w: [f32; 4], b: [f32; 2]) -> Vec<u8> {
        let tensor = |name: &str, dims: Vec<i64>, data: Vec<f32>| TensorProto { name: name.into(), dims, data_type: 1, float_data: data, ..Default::default() };
        let graph = GraphProto {
            name: "linear".into(),
            node: vec![node("MatMul", &["x", "w"], "h"), node("Add", &["h", "b"], "z"), node("Softmax", &["z"], "y")],
            initializer: vec![tensor("w", vec![2, 2], w.to_vec()), tensor("b", vec![2], b.to_vec())],
            input: vec![value("x", &[1, 2])],
            output: vec![value("y", &[1, 2])],
            ..Default::default()
        };
        ModelProto {
            ir_version: 7,
            opset_import: vec![OperatorSetIdProto { domain: String::new(), version: 13 }],
            graph: Some(graph),
            ..Default::default()
        }
        .encode_to_vec()
    }

    #[test]
    fn onnx_graphs_predict_probabilities() {
        let model = OnnxModel::load(&linear_softmax([1.0, -1.0, 0.0, 0.0], [0.0, 0.0]), vec![1, 0], 2).unwrap();
        // Columns pick event[1] then event[0]: z = [e1, -e1].
        let p = model.predict(&[5.0, 2.0]).unwrap();
        let expected = softmax(&[2.0, -2.0]);
        assert!((p[0] - expected[0]).abs() < 1e-6, "{p:?}");
        assert!(model.predict(&[f64::NAN, f64::NAN]).unwrap()[0] - 0.5 < 1e-6);
        assert!(OnnxModel::load(b"not onnx", vec![0, 1], 2).is_err());
    }
}
