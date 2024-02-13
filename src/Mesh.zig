const std = @import("std");
const math = @import("math.zig");
const Engine = @import("Engine.zig");
const vk = @import("vulkan-zig");

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
vertex_buffer: Engine.AllocatedBuffer,
