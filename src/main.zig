const std = @import("std");
const c = @import("c.zig");
const Engine = @import("Engine.zig");
const dispatch = @import("vk_dispatch.zig");

pub const vulkan_dispatch = struct {
    pub const device = dispatch.device;
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var engine = try Engine.init(allocator);
    defer engine.deinit();

    try engine.run();
    try engine.waitForIdle();
}
