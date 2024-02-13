const std = @import("std");
const math = @import("math.zig");
const Engine = @import("Engine.zig");
const vk = @import("vulkan-zig");
const c = @import("c.zig");

pub const VertexInputDescription = struct {
    bindings: std.ArrayList(vk.VertexInputBindingDescription),
    attributes: std.ArrayList(vk.VertexInputAttributeDescription),
    flags: vk.PipelineVertexInputStateCreateFlags = .{},
};

pub const Vertex = extern struct {
    position: math.Vec3,
    normal: math.Vec3,
    color: math.Vec3,

    pub fn getVertexDescription(allocator: std.mem.Allocator) !VertexInputDescription {
        const main_binding = vk.VertexInputBindingDescription{
            .binding = 0,
            .stride = @sizeOf(@This()),
            .input_rate = .vertex,
        };

        const attrs = [_]vk.VertexInputAttributeDescription{
            .{ .binding = 0, .location = 0, .format = .r32g32b32_sfloat, .offset = @offsetOf(@This(), "position") },
            .{ .binding = 0, .location = 1, .format = .r32g32b32_sfloat, .offset = @offsetOf(@This(), "normal") },
            .{ .binding = 0, .location = 2, .format = .r32g32b32_sfloat, .offset = @offsetOf(@This(), "color") },
        };

        var bindings = try std.ArrayList(vk.VertexInputBindingDescription).initCapacity(allocator, 1);
        var attributes = try std.ArrayList(vk.VertexInputAttributeDescription).initCapacity(allocator, attrs.len);

        try bindings.append(main_binding);
        try attributes.appendSlice(&attrs);

        return .{
            .bindings = bindings,
            .attributes = attributes,
        };
    }
};

vertices: std.ArrayList(Vertex),
vertex_buffer: Engine.AllocatedBuffer = undefined,

pub fn loadFromFile(allocator: std.mem.Allocator, filename: [*:0]const u8) !std.ArrayList(Vertex) {
    const flags = c.TINYOBJ_FLAG_TRIANGULATE;
    var attrib: c.tinyobj_attrib_t = undefined;
    var shapes: [*c]c.tinyobj_shape_t = undefined;
    var num_shapes: usize = undefined;
    var materials: [*c]c.tinyobj_material_t = undefined;
    var num_materials: usize = undefined;
    const res = c.tinyobj_parse_obj(
        &attrib,
        &shapes,
        &num_shapes,
        &materials,
        &num_materials,
        filename,
        fileReader,
        null,
        flags,
    );

    if (res != c.TINYOBJ_SUCCESS) return error.FailedToLoadModel;

    var unique_vertices = std.AutoHashMap(c.tinyobj_vertex_index_t, u32).init(allocator);
    defer unique_vertices.deinit();

    var vertices = std.ArrayList(Vertex).init(allocator);
    errdefer vertices.deinit();
    // var indices = std.ArrayList(u32).init(allocator);
    // errdefer indices.deinit();

    var face_offset: usize = 0;
    for (0..attrib.num_face_num_verts) |i| {
        std.debug.assert(attrib.face_num_verts[i] == 3);

        for (0..@intCast(attrib.face_num_verts[i])) |f| {
            const idx = attrib.faces[face_offset + f];

            var vertex: Vertex = undefined;

            vertex.position = .{
                attrib.vertices[3 * @as(usize, @intCast(idx.v_idx)) + 0],
                attrib.vertices[3 * @as(usize, @intCast(idx.v_idx)) + 1],
                attrib.vertices[3 * @as(usize, @intCast(idx.v_idx)) + 2],
            };

            if (attrib.num_normals >= 0 and idx.vn_idx >= 0) {
                vertex.normal = .{
                    attrib.normals[3 * @as(usize, @intCast(idx.vn_idx)) + 0],
                    attrib.normals[3 * @as(usize, @intCast(idx.vn_idx)) + 1],
                    attrib.normals[3 * @as(usize, @intCast(idx.vn_idx)) + 2],
                };
            }

            // if (attrib.num_texcoords >= 0 and idx.vt_idx >= 0) {
            //     vertex.uv = .{
            //         attrib.texcoords[2 * @as(usize, @intCast(idx.vt_idx)) + 0],
            //         attrib.texcoords[2 * @as(usize, @intCast(idx.vt_idx)) + 1],
            //     };
            // }

            if (attrib.material_ids[i] >= 0) {
                const mat_idx: usize = @intCast(attrib.material_ids[i]);
                vertex.color = .{
                    materials[mat_idx].diffuse[0],
                    materials[mat_idx].diffuse[1],
                    materials[mat_idx].diffuse[2],
                };
            } else {
                vertex.color = .{ 0.5, 0.5, 0.5 };
            }

            vertex.color = vertex.normal;

            // var index: u32 = undefined;
            // const entry = unique_vertices.get(idx);
            // if (entry) |value| {
            // index = value;
            // } else {
            // index = @intCast(vertices.items.len);
            // try unique_vertices.putNoClobber(idx, index);
            try vertices.append(vertex);
            // }

            // try indices.append(index);
        }

        face_offset += @intCast(attrib.face_num_verts[i]);
    }

    c.tinyobj_attrib_free(&attrib);
    c.tinyobj_shapes_free(shapes, num_shapes);
    c.tinyobj_materials_free(materials, num_materials);

    return vertices;
}

fn fileReader(
    _: ?*anyopaque,
    filename: [*c]const u8,
    _: c_int,
    _: [*c]const u8,
    buf: [*c][*c]u8,
    len: [*c]usize,
) callconv(.C) void {
    buf.* = null;
    len.* = 0;

    if (filename == null) return;

    const file = std.fs.cwd().openFileZ(filename, .{}) catch return;
    defer file.close();

    const content = file.readToEndAlloc(std.heap.page_allocator, std.math.maxInt(usize)) catch return;

    buf.* = content.ptr;
    len.* = content.len;
}
