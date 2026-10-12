//! Training datasets: an events table and a members table as Parquet, plus a
//! `dataset.json` manifest naming the feature set, taxonomy and class counts.
//! `squiggle-train` reads this layout. Missing feature values are nulls.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use serde::{Deserialize, Serialize};
use squiggle_core::Label;
use squiggle_core::features::FeatureSettings;

pub const FORMAT: &str = "heidtmare.squiggle.dataset/1";
pub const EVENTS_FILE: &str = "events.parquet";
pub const MEMBERS_FILE: &str = "members.parquet";
pub const MANIFEST_FILE: &str = "dataset.json";

#[derive(Clone, Debug, PartialEq)]
pub enum Values {
    Text(Vec<Option<String>>),
    /// NaN is written as null.
    Number(Vec<f64>),
}

impl Values {
    fn len(&self) -> usize {
        match self {
            Self::Text(v) => v.len(),
            Self::Number(v) => v.len(),
        }
    }
}

/// Named columns of equal length.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Table {
    pub columns: Vec<(String, Values)>,
}

impl Table {
    pub fn text(&mut self, name: &str, values: Vec<Option<String>>) {
        self.columns.push((name.to_string(), Values::Text(values)));
    }

    pub fn number(&mut self, name: &str, values: Vec<f64>) {
        self.columns.push((name.to_string(), Values::Number(values)));
    }

    pub fn rows(&self) -> usize {
        self.columns.first().map_or(0, |c| c.1.len())
    }

    pub fn column(&self, name: &str) -> Option<&Values> {
        self.columns.iter().find(|c| c.0 == name).map(|c| &c.1)
    }

    fn batch(&self, metadata: HashMap<String, String>) -> Result<RecordBatch, String> {
        let rows = self.rows();
        if let Some((name, _)) = self.columns.iter().find(|c| c.1.len() != rows) {
            return Err(format!("column {name} has a different length"));
        }
        let fields: Vec<Field> = self
            .columns
            .iter()
            .map(|(name, values)| {
                let kind = match values {
                    Values::Text(_) => DataType::Utf8,
                    Values::Number(_) => DataType::Float64,
                };
                Field::new(name, kind, true)
            })
            .collect();
        let arrays: Vec<ArrayRef> = self
            .columns
            .iter()
            .map(|(_, values)| -> ArrayRef {
                match values {
                    Values::Text(v) => Arc::new(StringArray::from(v.clone())),
                    Values::Number(v) => Arc::new(Float64Array::from(v.iter().map(|x| x.is_finite().then_some(*x)).collect::<Vec<_>>())),
                }
            })
            .collect();
        RecordBatch::try_new(Arc::new(Schema::new_with_metadata(fields, metadata)), arrays).map_err(|e| e.to_string())
    }
}

/// Writes `table` to `path` (Snappy-compressed), with `metadata` in the schema.
pub fn write_parquet(path: &Path, table: &Table, metadata: HashMap<String, String>) -> Result<(), String> {
    let batch = table.batch(metadata)?;
    let file = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let properties = WriterProperties::builder().set_compression(Compression::SNAPPY).build();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(properties)).map_err(|e| e.to_string())?;
    writer.write(&batch).map_err(|e| e.to_string())?;
    writer.close().map_err(|e| e.to_string())?;
    Ok(())
}

/// Reads a table written by [`write_parquet`]: text and float columns only, nulls as `None` and NaN.
pub fn read_parquet(path: &Path) -> Result<Table, String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file).and_then(|b| b.build()).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut table: Option<Table> = None;
    for batch in reader {
        let batch = batch.map_err(|e| format!("{}: {e}", path.display()))?;
        let table = table.get_or_insert_with(|| Table {
            columns: batch
                .schema()
                .fields()
                .iter()
                .map(|f| (f.name().clone(), if f.data_type() == &DataType::Utf8 { Values::Text(Vec::new()) } else { Values::Number(Vec::new()) }))
                .collect(),
        });
        for ((name, values), array) in table.columns.iter_mut().zip(batch.columns()) {
            match values {
                Values::Text(v) => {
                    let array = array.as_any().downcast_ref::<StringArray>().ok_or_else(|| format!("column {name} is not text"))?;
                    v.extend((0..array.len()).map(|i| (!array.is_null(i)).then(|| array.value(i).to_string())));
                }
                Values::Number(v) => {
                    let array = array.as_any().downcast_ref::<Float64Array>().ok_or_else(|| format!("column {name} is not a float"))?;
                    v.extend((0..array.len()).map(|i| if array.is_null(i) { f64::NAN } else { array.value(i) }));
                }
            }
        }
    }
    Ok(table.unwrap_or_default())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub project: String,
    pub created: String,
    pub analyst: String,
    pub feature_set: String,
    pub taxonomy_version: u32,
    pub labels: Vec<Label>,
    /// Class rejected events carry.
    pub noise_class: String,
    /// Settings the features were computed with.
    #[serde(default)]
    pub feature_settings: Option<FeatureSettings>,
    pub event_features: Vec<String>,
    pub member_features: Vec<String>,
    /// Events per class (`null` for unlabelled).
    pub classes: Vec<(Option<String>, usize)>,
    pub events: usize,
    pub members: usize,
    /// blake3 of each table file.
    pub hashes: Vec<(String, String)>,
}

/// Writes both tables and the manifest into `dir`. Fills the manifest's row counts and hashes.
pub fn write_dataset(dir: &Path, events: &Table, members: &Table, mut manifest: Manifest) -> Result<Manifest, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let metadata = HashMap::from([
        ("format".to_string(), FORMAT.to_string()),
        ("feature_set".to_string(), manifest.feature_set.clone()),
        ("taxonomy_version".to_string(), manifest.taxonomy_version.to_string()),
    ]);
    manifest.hashes.clear();
    for (name, table) in [(EVENTS_FILE, events), (MEMBERS_FILE, members)] {
        let path = dir.join(name);
        write_parquet(&path, table, metadata.clone())?;
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        manifest.hashes.push((name.to_string(), blake3::hash(&bytes).to_hex().to_string()));
    }
    manifest.events = events.rows();
    manifest.members = members.rows();
    let json = serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?;
    let path = dir.join(MANIFEST_FILE);
    std::fs::write(&path, json).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(manifest)
}

/// Reads a dataset written by [`write_dataset`], checking each table against its hash.
pub fn read_dataset(dir: &Path) -> Result<(Table, Table, Manifest), String> {
    let path = dir.join(MANIFEST_FILE);
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let manifest: Manifest = serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    if manifest.format != FORMAT {
        return Err(format!("format {} is not {FORMAT}", manifest.format));
    }
    let mut tables = Vec::new();
    for name in [EVENTS_FILE, MEMBERS_FILE] {
        let path = dir.join(name);
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let expected = manifest.hashes.iter().find(|h| h.0 == name).map(|h| h.1.as_str());
        if expected != Some(blake3::hash(&bytes).to_hex().as_str()) {
            return Err(format!("{name} does not match the manifest's hash"));
        }
        tables.push(read_parquet(&path)?);
    }
    let members = tables.pop().unwrap_or_default();
    let events = tables.pop().unwrap_or_default();
    Ok((events, members, manifest))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_round_trip_with_nulls_and_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let mut events = Table::default();
        events.text("label", vec![Some("explosion".into()), Some("noise".into()), None]);
        events.number("p_s_log", vec![0.5, f64::NAN, -0.4]);
        let mut members = Table::default();
        members.text("event", vec![Some("a".into())]);
        let manifest = Manifest {
            format: FORMAT.into(),
            project: "p".into(),
            created: "now".into(),
            analyst: "ana".into(),
            feature_set: "f/1".into(),
            taxonomy_version: 3,
            labels: squiggle_core::taxonomy::seed(),
            noise_class: squiggle_core::taxonomy::NOISE.into(),
            feature_settings: Some(FeatureSettings::default()),
            event_features: vec!["p_s_log".into()],
            member_features: vec![],
            classes: vec![],
            events: 0,
            members: 0,
            hashes: vec![],
        };
        let written = write_dataset(dir.path(), &events, &members, manifest).unwrap();
        assert_eq!((written.events, written.members, written.hashes.len()), (3, 1, 2));

        let file = File::open(dir.path().join(EVENTS_FILE)).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        assert_eq!(builder.schema().metadata().get("taxonomy_version").map(String::as_str), Some("3"));
        let batch = builder.build().unwrap().next().unwrap().unwrap();
        let labels = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!((labels.value(1), labels.is_null(2)), ("noise", true));
        let ratios = batch.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!((ratios.value(0), ratios.is_null(1)), (0.5, true));

        let json: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.path().join(MANIFEST_FILE)).unwrap()).unwrap();
        assert_eq!(json["format"], FORMAT);
        assert_eq!(json["labels"][1]["id"], "explosion");

        let (read, _, back) = read_dataset(dir.path()).unwrap();
        assert_eq!(back, written);
        assert_eq!(read.column("label"), events.column("label"));
        let Some(Values::Number(ratios)) = read.column("p_s_log") else { panic!() };
        assert_eq!((ratios[0], ratios[1].is_nan()), (0.5, true));
        std::fs::write(dir.path().join(MEMBERS_FILE), b"x").unwrap();
        assert!(read_dataset(dir.path()).unwrap_err().contains("hash"));

        let mut ragged = Table::default();
        ragged.number("a", vec![1.0]);
        ragged.number("b", vec![]);
        assert!(write_parquet(&dir.path().join("x.parquet"), &ragged, HashMap::new()).is_err());
    }
}
