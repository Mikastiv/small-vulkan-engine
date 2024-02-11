const std = @import("std");
const c = @import("c.zig");
const Window = @import("Window.zig");
const Allocator = std.mem.Allocator;

const window_width = 1700;
const window_height = 900;
const window_title = "Vulkan Engine";

var is_initialized: bool = false;
var allocator: Allocator = undefined;
var window: *Window = undefined;
var frame_number: usize = 0;
var stop_rendering: bool = false;

pub fn init(alloc: Allocator) !void {
    std.debug.assert(!is_initialized);

    allocator = alloc;

    if (c.glfwInit() == c.GLFW_FALSE) return error.GlfwInitFailed;
    errdefer c.glfwTerminate();

    _ = c.glfwSetErrorCallback(errorCallback);

    window = try Window.init(allocator, window_width, window_height, window_title);
    errdefer window.deinit(allocator);

    is_initialized = true;
}

pub fn deinit() void {
    std.debug.assert(is_initialized);

    window.deinit(allocator);
    c.glfwTerminate();
}

pub fn run() !void {
    while (!window.shouldClose()) {
        c.glfwPollEvents();

        if (window.minimized) {
            stop_rendering = true;
        } else {
            stop_rendering = false;
        }

        if (stop_rendering) {
            std.time.sleep(std.time.ns_per_ms * 100);
            continue;
        }

        try draw();
    }
}

fn draw() !void {}

fn errorCallback(error_code: i32, description: [*c]const u8) callconv(.C) void {
    const log = std.log.scoped(.glfw);
    log.err("{}: {s}\n", .{ error_code, description });
}
