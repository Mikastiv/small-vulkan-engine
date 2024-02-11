const std = @import("std");
const c = @import("c.zig");
const Window = @import("Window.zig");
const Allocator = std.mem.Allocator;
const vk = @import("vulkan-zig");
const vkk = @import("vk-kickstart");

const vki = vkk.vki;
const vkd = vkk.vkd;

const window_width = 1700;
const window_height = 900;
const window_title = "Vulkan Engine";

allocator: Allocator,
window: *Window,
frame_number: usize = 0,
stop_rendering: bool = false,
instance: vkk.Instance,
device: vkk.Device,
surface: vk.SurfaceKHR,

pub fn init(allocator: Allocator) !@This() {
    if (c.glfwInit() == c.GLFW_FALSE) return error.GlfwInitFailed;
    errdefer c.glfwTerminate();

    _ = c.glfwSetErrorCallback(errorCallback);

    const window = try Window.init(allocator, window_width, window_height, window_title);
    errdefer window.deinit(allocator);

    const instance = try vkk.Instance.create(allocator, c.glfwGetInstanceProcAddress, .{
        .required_api_version = vk.API_VERSION_1_3,
    });
    errdefer instance.destroy();

    const surface = try window.createSurface(instance.handle);
    defer vki().destroySurfaceKHR(instance.handle, surface, instance.allocation_callbacks);

    const physical_device = try vkk.PhysicalDevice.select(allocator, &instance, .{
        .surface = surface,
        .required_api_version = vk.API_VERSION_1_3,
        .required_features_12 = .{
            .buffer_device_address = vk.TRUE,
            .descriptor_indexing = vk.TRUE,
        },
        .required_features_13 = .{
            .dynamic_rendering = vk.TRUE,
            .synchronization_2 = vk.TRUE,
        },
    });

    const device = try vkk.Device.create(allocator, &physical_device, null);
    errdefer device.destroy();

    return .{
        .allocator = allocator,
        .window = window,
        .instance = instance,
        .device = device,
        .surface = surface,
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
    log.err("glfw: {d}: {s}\n", .{ error_code, description });
}
