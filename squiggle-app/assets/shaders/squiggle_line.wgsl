// Expands a polyline into screen-space ribbon geometry.
//
// Each plotted point contributes two vertices, one either side of the line.
// Widening happens after projection, in pixels, so a squiggle keeps the same
// visual thickness however far the plot is zoomed -- and, because the plot
// scales x and y independently, that is something a pre-widened mesh could
// not do.

#import bevy_sprite::{
    mesh2d_functions::{get_world_from_local, mesh2d_position_local_to_clip},
    mesh2d_view_bindings::view,
}

struct SquiggleLine {
    color: vec4<f32>,
    // Half the stroke width, in physical pixels.
    half_width: f32,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> material: SquiggleLine;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    // The neighbouring points, used to work out the joint angle.
    @location(1) previous: vec2<f32>,
    @location(2) next: vec2<f32>,
    // -1 or +1: which side of the line this vertex is pushed towards.
    @location(3) side: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

// Clip space to pixels, relative to the centre of the viewport.
fn to_pixels(clip: vec4<f32>, viewport: vec2<f32>) -> vec2<f32> {
    return (clip.xy / clip.w) * 0.5 * viewport;
}

const EPSILON: f32 = 1e-6;
// Stops a hairpin turn from firing the joint off to infinity.
const MITER_LIMIT: f32 = 4.0;

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let world_from_local = get_world_from_local(vertex.instance_index);
    let clip = mesh2d_position_local_to_clip(world_from_local, vec4(vertex.position, 1.0));
    let clip_previous = mesh2d_position_local_to_clip(
        world_from_local, vec4(vertex.previous, vertex.position.z, 1.0));
    let clip_next = mesh2d_position_local_to_clip(
        world_from_local, vec4(vertex.next, vertex.position.z, 1.0));

    let viewport = view.viewport.zw;
    let here = to_pixels(clip, viewport);
    let behind = to_pixels(clip_previous, viewport);
    let ahead = to_pixels(clip_next, viewport);

    let incoming = here - behind;
    let outgoing = ahead - here;
    let incoming_length = length(incoming);
    let outgoing_length = length(outgoing);

    // At the ends of the line there is only one direction to go on.
    var direction_in = select(vec2(0.0), incoming / incoming_length, incoming_length > EPSILON);
    var direction_out = select(vec2(0.0), outgoing / outgoing_length, outgoing_length > EPSILON);
    if (incoming_length <= EPSILON) { direction_in = direction_out; }
    if (outgoing_length <= EPSILON) { direction_out = direction_in; }

    // Degenerate input: both neighbours coincide with this point.
    if (length(direction_out) <= EPSILON) {
        direction_in = vec2(1.0, 0.0);
        direction_out = vec2(1.0, 0.0);
    }

    // The miter bisects the joint. Where the line doubles back on itself the
    // bisector vanishes, so fall back to a square end.
    var bisector = direction_in + direction_out;
    if (length(bisector) <= EPSILON) {
        bisector = direction_out;
    }
    bisector = normalize(bisector);

    let miter = vec2(-bisector.y, bisector.x);
    let segment_normal = vec2(-direction_out.y, direction_out.x);
    // Sharper joints need a longer miter to keep the outer edge continuous.
    let projection = max(abs(dot(miter, segment_normal)), 1.0 / MITER_LIMIT);

    let offset = miter * (vertex.side * material.half_width / projection);
    let pixels = here + offset;

    var out: VertexOutput;
    out.clip_position = vec4(pixels / (0.5 * viewport) * clip.w, clip.z, clip.w);
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return material.color;
}
