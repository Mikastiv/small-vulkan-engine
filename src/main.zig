const std = @import("std");
const c = @import("c.zig");
const Window = @import("Window.zig");

const window_width = 800;
const window_height = 600;

pub fn main() !void {
    if (c.glfwInit() == c.GLFW_FALSE) return error.GlfwInitFailed;
    defer c.glfwTerminate();

    _ = c.glfwSetErrorCallback(errorCallback);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    const window = try Window.init(allocator, window_width, window_height, "vulkan guide");
    defer window.deinit(allocator);

    while (!window.shouldClose()) {
        c.glfwPollEvents();
    }
}

fn errorCallback(error_code: i32, description: [*c]const u8) callconv(.C) void {
    const log = std.log.scoped(.glfw);
    log.err("{}: {s}\n", .{ error_code, description });
}
