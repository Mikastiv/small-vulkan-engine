const std = @import("std");
const c = @import("c.zig");
const engine = @import("engine.zig");

const window_width = 800;
const window_height = 600;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    try engine.init(allocator);
    defer engine.deinit();

    try engine.run();
}
