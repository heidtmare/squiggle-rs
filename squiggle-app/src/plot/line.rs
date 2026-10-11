//! Turning a squiggle's points into GPU geometry.
//!
//! The Java renderer walked every coordinate on the CPU each frame, projecting
//! and culling as it went. Here the points are uploaded once as a mesh and the
//! projection happens in the vertex shader, so panning and zooming cost nothing
//! per point.

use bevy::asset::{Asset, RenderAssetUsages};
use bevy::mesh::{Indices, MeshVertexAttribute, MeshVertexBufferLayoutRef, PrimitiveTopology};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError, VertexFormat,
};
use bevy::shader::ShaderRef;
use bevy::sprite_render::{AlphaMode2d, Material2d, Material2dKey};
use squiggle_core::{Point, Rgba};

pub const SHADER_PATH: &str = "shaders/squiggle_line.wgsl";

/// The point before this one, so the shader can measure the joint angle.
pub const ATTRIBUTE_PREVIOUS: MeshVertexAttribute =
    MeshVertexAttribute::new("SquigglePrevious", 0x5175_9001, VertexFormat::Float32x2);

/// The point after this one.
pub const ATTRIBUTE_NEXT: MeshVertexAttribute =
    MeshVertexAttribute::new("SquiggleNext", 0x5175_9002, VertexFormat::Float32x2);

/// Which side of the line this vertex sits on: -1 or +1.
pub const ATTRIBUTE_SIDE: MeshVertexAttribute =
    MeshVertexAttribute::new("SquiggleSide", 0x5175_9003, VertexFormat::Float32);

/// Uniform block matching `SquiggleLine` in the shader.
#[derive(Clone, Copy, Debug, ShaderType)]
pub struct LineUniform {
    pub color: LinearRgba,
    /// Half the stroke width in *physical* pixels, which is the unit the
    /// shader's viewport rectangle is expressed in.
    pub half_width: f32,
}

/// Paints a squiggle as a constant-width screen-space line.
#[derive(Asset, AsBindGroup, TypePath, Clone, Debug)]
pub struct SquiggleLineMaterial {
    #[uniform(0)]
    pub line: LineUniform,
}

impl SquiggleLineMaterial {
    pub fn new(color: Rgba, width_logical: f32, scale_factor: f32) -> Self {
        Self {
            line: LineUniform {
                // The model stores sRGB; the GPU wants linear.
                color: Color::srgba(color.red, color.green, color.blue, color.alpha).to_linear(),
                half_width: (width_logical * scale_factor / 2.0).max(0.5),
            },
        }
    }
}

impl Material2d for SquiggleLineMaterial {
    fn vertex_shader() -> ShaderRef {
        SHADER_PATH.into()
    }

    fn fragment_shader() -> ShaderRef {
        SHADER_PATH.into()
    }

    fn alpha_mode(&self) -> AlphaMode2d {
        // Blending puts squiggles in the same sorted phase as the gizmo
        // overlay, so plain z ordering decides what covers what.
        AlphaMode2d::Blend
    }

    fn specialize(
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: Material2dKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.vertex.buffers = vec![layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            ATTRIBUTE_PREVIOUS.at_shader_location(1),
            ATTRIBUTE_NEXT.at_shader_location(2),
            ATTRIBUTE_SIDE.at_shader_location(3),
        ])?];
        // A ribbon flips winding whenever the line turns back on itself.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

/// Builds the ribbon for `points` in local space.
///
/// Each point becomes two vertices, one per side, carrying its neighbours so
/// the shader can miter the joints. Non-finite points break the line. Returns
/// `None` when there is no segment to draw.
pub fn build_mesh(points: &[Point]) -> Option<Mesh> {
    let finite = |p: &Point| p.x.is_finite() && p.y.is_finite();
    let runs: Vec<&[Point]> = points.split(|p| !finite(p)).filter(|run| run.len() >= 2).collect();
    let point_count: usize = runs.iter().map(|run| run.len()).sum();
    if point_count == 0 {
        return None;
    }

    let vertex_count = point_count * 2;
    let mut positions = Vec::with_capacity(vertex_count);
    let mut previous = Vec::with_capacity(vertex_count);
    let mut next = Vec::with_capacity(vertex_count);
    let mut sides = Vec::with_capacity(vertex_count);
    let mut indices = Vec::with_capacity((point_count - runs.len()) * 6);

    for run in runs {
        let base = positions.len() as u32;
        for (index, point) in run.iter().enumerate() {
            let position = [point.x as f32, point.y as f32, 0.0];
            // The ends have no neighbour on one side; repeating the point itself
            // tells the shader to square the line off there.
            let before = run[index.saturating_sub(1)];
            let after = run[(index + 1).min(run.len() - 1)];

            for side in [-1.0f32, 1.0] {
                positions.push(position);
                previous.push([before.x as f32, before.y as f32]);
                next.push([after.x as f32, after.y as f32]);
                sides.push(side);
            }
        }
        // Two triangles bridge each pair of adjacent points.
        for segment in 0..run.len() as u32 - 1 {
            let at = base + segment * 2;
            indices.extend_from_slice(&[at, at + 1, at + 2, at + 1, at + 3, at + 2]);
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        // The mesh is rebuilt from the model when points change and is never
        // read back, so it does not need to stay in main memory.
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(ATTRIBUTE_PREVIOUS, previous);
    mesh.insert_attribute(ATTRIBUTE_NEXT, next);
    mesh.insert_attribute(ATTRIBUTE_SIDE, sides);
    mesh.insert_indices(Indices::U32(indices));
    Some(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn points(count: usize) -> Vec<Point> {
        (0..count).map(|i| Point::new(i as f64, (i * i) as f64)).collect()
    }

    #[test]
    fn a_single_point_is_not_a_line() {
        assert!(build_mesh(&points(1)).is_none());
        assert!(build_mesh(&[]).is_none());
    }

    #[test]
    fn every_point_contributes_both_sides() {
        let mesh = build_mesh(&points(5)).expect("five points make a line");
        assert_eq!(mesh.count_vertices(), 10);
    }

    #[test]
    fn each_segment_gets_two_triangles() {
        let mesh = build_mesh(&points(5)).expect("five points make a line");
        let Some(Indices::U32(indices)) = mesh.indices() else {
            panic!("expected u32 indices");
        };
        assert_eq!(indices.len(), 4 * 6);
        assert!(indices.iter().all(|i| (*i as usize) < mesh.count_vertices()));
    }

    #[test]
    fn non_finite_points_break_the_line() {
        let mut pts = points(6);
        pts[2].y = f64::NAN;
        // Runs [0, 1] and [3, 4, 5]: three segments, isolated points dropped.
        let mesh = build_mesh(&pts).expect("two runs");
        assert_eq!(mesh.count_vertices(), 10);
        let Some(Indices::U32(indices)) = mesh.indices() else {
            panic!("expected u32 indices");
        };
        assert_eq!(indices.len(), 3 * 6);
        pts[4].x = f64::INFINITY;
        assert!(build_mesh(&pts[..3]).is_some());
        assert!(build_mesh(&[pts[0], Point::new(f64::NAN, 0.0), pts[1]]).is_none());
    }

    #[test]
    fn line_ends_repeat_themselves_as_their_own_neighbour() {
        let pts = points(3);
        let mesh = build_mesh(&pts).expect("three points make a line");

        let Some(bevy::mesh::VertexAttributeValues::Float32x2(previous)) =
            mesh.attribute(ATTRIBUTE_PREVIOUS)
        else {
            panic!("expected the previous-point attribute");
        };
        // Both vertices of the first point look back at the first point.
        assert_eq!(previous[0], [pts[0].x as f32, pts[0].y as f32]);
        assert_eq!(previous[1], [pts[0].x as f32, pts[0].y as f32]);
        // Later points look back at their real predecessor.
        assert_eq!(previous[2], [pts[0].x as f32, pts[0].y as f32]);
        assert_eq!(previous[4], [pts[1].x as f32, pts[1].y as f32]);
    }
}
