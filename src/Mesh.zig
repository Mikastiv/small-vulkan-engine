const std = @import("std");
const math = @import("math.zig");
const Engine = @import("Engine.zig");
const vk = @import("vulkan-zig");
const obj_loader = @import("obj_loader.zig");

pub const VertexInputDescription = struct {
    bindings: std.ArrayList(vk.VertexInputBindingDescription),
    attributes: std.ArrayList(vk.VertexInputAttributeDescription),
    flags: vk.PipelineVertexInputStateCreateFlags = .{},
};

pub const Vertex = extern struct {
    position: math.Vec3,
    normal: math.Vec3,
    color: math.Vec3,
    uv: math.Vec2,

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
            .{ .binding = 0, .location = 3, .format = .r32g32_sfloat, .offset = @offsetOf(@This(), "uv") },
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
vertex_buffer: Engine.Buffer = undefined,

pub fn loadFromFile(allocator: std.mem.Allocator, filename: []const u8) !std.ArrayList(Vertex) {
    var mesh = try obj_loader.parse_file(allocator, filename);
    defer mesh.deinit();

    var vertices = std.ArrayList(Vertex).init(allocator);
    errdefer vertices.deinit();

    var i: u32 = 0;
    for (mesh.objects) |object| {
        var index_count: usize = 0;
        for (object.face_vertices) |face_vx_count| {
            if (face_vx_count < 3) {
                @panic("Face has fewer than 3 vertices. Not a valid polygon.");
            }

            for (0..face_vx_count) |vx_index| {
                const obj_index = object.indices[index_count];
                const pos = mesh.vertices[obj_index.vertex];
                const nml = mesh.normals[obj_index.normal];
                const uvs = mesh.uvs[obj_index.uv];

                const vx = Vertex{
                    .position = .{ pos[0], pos[1], pos[2] },
                    .normal = .{ nml[0], nml[1], nml[2] },
                    .color = .{ nml[0], nml[1], nml[2] },
                    .uv = .{ uvs[0], 1.0 - uvs[1] },
                };

                // Triangulate the polygon
                if (vx_index > 2) {
                    const v0 = vertices.items[vertices.items.len - 3];
                    const v1 = vertices.items[vertices.items.len - 1];
                    try vertices.appendSlice(&.{ v0, v1 });
                }

                try vertices.append(vx);

                index_count += 1;
                i += 1;
            }
        }
    }

    return vertices;
}
