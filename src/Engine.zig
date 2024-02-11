const std = @import("std");
const c = @import("c.zig");
const Window = @import("Window.zig");
const Allocator = std.mem.Allocator;

const window_width = 1700;
const window_height = 900;
const window_title = "Vulkan Engine";

allocator: Allocator,
window: *Window,
frame_number: usize = 0,
stop_rendering: bool = false,

pub fn init(allocator: Allocator) !@This() {
    if (c.glfwInit() == c.GLFW_FALSE) return error.GlfwInitFailed;
    errdefer c.glfwTerminate();

    _ = c.glfwSetErrorCallback(errorCallback);

    const window = try Window.init(allocator, window_width, window_height, window_title);
    errdefer window.deinit(allocator);

    return .{
        .allocator = allocator,
        .window = window,
    };
}

pub fn deinit(self: @This()) void {
    self.window.deinit(self.allocator);
    c.glfwTerminate();
}

pub fn run(self: *@This()) !void {
    while (!self.window.shouldClose()) {
        c.glfwPollEvents();

        if (self.window.minimized) {
            self.stop_rendering = true;
        } else {
            self.stop_rendering = false;
        }

        if (self.stop_rendering) {
            std.time.sleep(std.time.ns_per_ms * 100);
            continue;
        }

        try self.draw();
    }
}

fn draw(self: *@This()) !void {
    _ = self;
}

fn errorCallback(error_code: i32, description: [*c]const u8) callconv(.C) void {
    const log = std.log.scoped(.glfw);
    log.err("{}: {s}\n", .{ error_code, description });
}
