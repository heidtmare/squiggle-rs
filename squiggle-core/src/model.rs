//! Imported model packages, by reference. Weights stay on disk; the project
//! records where and which bytes, so a changed package is detected.

/// One imported model package.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ModelRef {
    /// `heidtmare.squiggle.model/<name>@<version>`; unique within a project.
    pub id: String,
    /// Package directory.
    pub path: String,
    /// blake3 of the package files, at import.
    pub hash: String,
    pub enabled: bool,
}
