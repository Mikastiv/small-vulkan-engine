pub usingnamespace @cImport({
    @cInclude("GLFW/glfw3.h");
    @cInclude("stb_image.h");
    @cInclude("vk_mem_alloc.h");
    @cInclude("cimgui.h");
    @cInclude("cimgui_impl_vulkan.h");
    @cInclude("cimgui_impl_glfw.h");
});

const vk = @import("vulkan-zig");

pub extern fn glfwGetInstanceProcAddress(instance: vk.Instance, procname: [*:0]const u8) vk.PfnVoidFunction;
pub extern fn glfwCreateWindowSurface(instance: vk.Instance, window: *@This().GLFWwindow, allocation_callbacks: ?*const vk.AllocationCallbacks, surface: *vk.SurfaceKHR) vk.Result;
