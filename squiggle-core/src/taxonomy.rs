//! Event labels: a hierarchy analysts extend. Models trained on a parent label
//! still apply to its subtypes. Rejected events export as [`NOISE`], which is
//! not a label.

use core::fmt;

use crate::phenomenology::PhenomenologyId;

/// Class name rejected events export as.
pub const NOISE: &str = "noise";

/// Stable slug, e.g. `explosion`. Unique within a project.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct LabelId(pub String);

impl LabelId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for LabelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Label {
    pub id: LabelId,
    pub name: String,
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Option::is_none"))]
    pub parent: Option<LabelId>,
    /// Phenomenologies an event of this class is expected to show.
    #[cfg_attr(feature = "serde", serde(default, skip_serializing_if = "Vec::is_empty"))]
    pub expects: Vec<PhenomenologyId>,
}

pub const EARTHQUAKE: &str = "earthquake";
pub const EXPLOSION: &str = "explosion";
pub const OTHER: &str = "other";

/// The seed taxonomy.
pub fn seed() -> Vec<Label> {
    let label = |id: &str, name: &str, expects: &[&str]| Label {
        id: LabelId::new(id),
        name: name.to_string(),
        parent: None,
        expects: expects.iter().map(|p| PhenomenologyId::new(*p)).collect(),
    };
    vec![
        label(EARTHQUAKE, "Earthquake", &["seismic"]),
        label(EXPLOSION, "Explosion", &["seismic", "infrasound", "acoustic"]),
        label(OTHER, "Other", &[]),
    ]
}

/// Labels in a project, with a version bumped on every definition.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Taxonomy {
    pub labels: Vec<Label>,
    pub version: u32,
}

impl Taxonomy {
    pub fn get(&self, id: &LabelId) -> Option<&Label> {
        self.labels.iter().find(|l| &l.id == id)
    }

    /// `id` and its ancestors, nearest first.
    pub fn lineage<'a>(&'a self, id: &'a LabelId) -> impl Iterator<Item = &'a LabelId> + 'a {
        let mut next = Some(id);
        core::iter::from_fn(move || {
            let current = next?;
            next = self.get(current).and_then(|l| l.parent.as_ref());
            Some(current)
        })
        .take(self.labels.len() + 1)
    }

    /// Whether `id` is `ancestor` or descends from it.
    pub fn is_a(&self, id: &LabelId, ancestor: &LabelId) -> bool {
        self.lineage(id).any(|l| l == ancestor)
    }

    /// Display name with ancestors, e.g. "Explosion › Quarry".
    pub fn path(&self, id: &LabelId) -> String {
        let mut names: Vec<&str> = self.lineage(id).map(|l| self.get(l).map_or(l.as_str(), |l| l.name.as_str())).collect();
        names.reverse();
        names.join(" › ")
    }

    /// Labels in tree order (each parent before its children); keys `1`–`9` follow it.
    pub fn ordered(&self) -> Vec<&Label> {
        fn visit<'a>(t: &'a Taxonomy, parent: Option<&LabelId>, out: &mut Vec<&'a Label>) {
            for label in t.labels.iter().filter(|l| l.parent.as_ref() == parent) {
                out.push(label);
                visit(t, Some(&label.id), out);
            }
        }
        let mut out = Vec::new();
        visit(self, None, &mut out);
        out
    }

    /// Validates a definition: a known parent and no cycle through it.
    pub fn check(&self, label: &Label) -> Result<(), String> {
        if label.id.as_str().is_empty() || label.id.as_str() == NOISE {
            return Err(format!("\"{}\" is not a usable label id", label.id));
        }
        if let Some(parent) = &label.parent {
            if self.get(parent).is_none() {
                return Err(format!("no parent label {parent}"));
            }
            if self.lineage(parent).any(|l| l == &label.id) {
                return Err(format!("{} cannot descend from itself", label.id));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn taxonomy() -> Taxonomy {
        let mut labels = seed();
        labels.push(Label { id: LabelId::new("quarry"), name: "Quarry".into(), parent: Some(LabelId::new(EXPLOSION)), expects: vec![] });
        Taxonomy { labels, version: 4 }
    }

    #[test]
    fn subtypes_descend_from_their_parent() {
        let t = taxonomy();
        let quarry = LabelId::new("quarry");
        assert!(t.is_a(&quarry, &LabelId::new(EXPLOSION)));
        assert!(!t.is_a(&quarry, &LabelId::new(EARTHQUAKE)));
        assert_eq!(t.path(&quarry), "Explosion › Quarry");
        let order: Vec<&str> = t.ordered().iter().map(|l| l.id.as_str()).collect();
        assert_eq!(order, [EARTHQUAKE, EXPLOSION, "quarry", OTHER]);
    }

    #[test]
    fn cycles_and_unknown_parents_are_refused() {
        let t = taxonomy();
        let explosion = Label { parent: Some(LabelId::new("quarry")), ..t.get(&LabelId::new(EXPLOSION)).unwrap().clone() };
        assert!(t.check(&explosion).is_err());
        let orphan = Label { id: LabelId::new("x"), name: "X".into(), parent: Some(LabelId::new("nope")), expects: vec![] };
        assert!(t.check(&orphan).is_err());
        assert!(t.check(&Label { id: LabelId::new(NOISE), ..orphan }).is_err());
    }
}
