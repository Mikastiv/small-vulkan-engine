pub usingnamespace @cImport({
    @cInclude("GLFW/glfw3.h");
    @cInclude("stb_image.h");
    @cInclude("cimgui.h");
    @cInclude("cimgui_impl_vulkan.h");
    @cInclude("cimgui_impl_glfw.h");
    @cInclude("tiny_obj_loader.h");
});

const vk = @import("vulkan-zig");

pub extern fn glfwGetInstanceProcAddress(instance: vk.Instance, procname: [*:0]const u8) vk.PfnVoidFunction;
pub extern fn glfwCreateWindowSurface(instance: vk.Instance, window: *@This().GLFWwindow, allocation_callbacks: ?*const vk.AllocationCallbacks, surface: *vk.SurfaceKHR) vk.Result;

pub fn vulkanZigHandleToC(comptime T: type, zig_handle: anytype) T {
    const Z = @typeInfo(@TypeOf(zig_handle));
    if (Z != .Enum) @compileError("must be a Vulkan handle");

    const handle_int: Z.Enum.tag_type = @intFromEnum(zig_handle);
    const handle_c: T = @ptrFromInt(handle_int);

    return handle_c;
}

pub fn vulkanCHandleToZig(comptime T: type, c_handle: anytype) T {
    const Z = @typeInfo(@TypeOf(c_handle));
    if (Z != .Optional) @compileError("must be a Vulkan handle");
    if (@typeInfo(Z.Optional.child) != .Pointer) @compileError("must be a Vulkan handle");

    const handle_int: u64 = @intFromPtr(c_handle);
    const handle_zig: T = @enumFromInt(handle_int);

    return handle_zig;
}
