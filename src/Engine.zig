const std = @import("std");
const c = @import("c.zig");
const Window = @import("Window.zig");
const Allocator = std.mem.Allocator;
const vk = @import("vulkan-zig");
const vkk = @import("vk-kickstart");
const Shaders = @import("shaders");
const vk_init = @import("vk_init.zig");
const Mesh = @import("Mesh.zig");
const math = @import("math.zig");

const vki = vkk.dispatch.vki;
const vkd = vkk.dispatch.vkd;
const DeviceDispatch = vkk.dispatch.DeviceDispatch;

const log = std.log.scoped(.engine);

const window_width = 1700;
const window_height = 900;
const window_title = "Vulkan Engine";

pub const AllocatedBuffer = struct {
    buffer: vk.Buffer,
    allocation: c.VmaAllocation,
};

pub const MeshPushConstants = extern struct {
    data: math.Vec4 align(16),
    render_matrix: math.Mat4 align(16),
};

allocator: Allocator,
vma: c.VmaAllocator,
window: *Window,
frame_number: usize = 0,
stop_rendering: bool = false,
instance: vkk.Instance,
device: vkk.Device,
deletion_queue: std.ArrayList(VulkanDeleter),
buffer_deletion_queue: std.ArrayList(AllocatedBuffer),
surface: vk.SurfaceKHR,
swapchain: vkk.Swapchain,
swapchain_images: []vk.Image,
swapchain_image_views: []vk.ImageView,
command_pool: vk.CommandPool,
main_command_buffer: vk.CommandBuffer,
render_pass: vk.RenderPass,
framebuffers: []vk.Framebuffer,
render_fence: vk.Fence,
present_semaphore: vk.Semaphore,
render_semaphore: vk.Semaphore,
triangle_pipeline_layout: vk.PipelineLayout,
triangle_pipeline: vk.Pipeline,
red_triangle_pipeline: vk.Pipeline,
mesh_pipeline_layout: vk.PipelineLayout,
mesh_pipeline: vk.Pipeline,
triangle_mesh: Mesh,
selected_shader: u32 = 0,

pub fn init(allocator: Allocator) !@This() {
    if (c.glfwInit() == c.GLFW_FALSE) return error.GlfwInitFailed;

    _ = c.glfwSetErrorCallback(errorCallback);

    const window = try Window.init(allocator, window_width, window_height, window_title);

    const instance = try vkk.Instance.create(allocator, c.glfwGetInstanceProcAddress, .{});
    const surface = try window.createSurface(instance.handle);
    const physical_device = try vkk.PhysicalDevice.select(allocator, &instance, .{
        .surface = surface,
    });
    const device = try vkk.Device.create(allocator, &physical_device, null);

    const vma_info = c.VmaAllocatorCreateInfo{
        .instance = c.vulkanZigHandleToC(c.VkInstance, instance.handle),
        .physicalDevice = c.vulkanZigHandleToC(c.VkPhysicalDevice, physical_device.handle),
        .device = c.vulkanZigHandleToC(c.VkDevice, device.handle),
    };
    var vma: c.VmaAllocator = undefined;
    try vkCheck(c.vmaCreateAllocator(&vma_info, &vma));

    var deletion_queue = std.ArrayList(VulkanDeleter).init(allocator);

    const swapchain = try vkk.Swapchain.create(allocator, &device, surface, .{
        .desired_extent = window.extent(),
        .desired_present_modes = &.{
            .fifo_khr,
        },
    });
    try deletion_queue.append(VulkanDeleter.make(swapchain.handle, DeviceDispatch.destroySwapchainKHR));

    const images = try swapchain.getImages(allocator);

    const image_views = try swapchain.getImageViews(allocator, images);
    for (image_views) |view| {
        try deletion_queue.append(VulkanDeleter.make(view, DeviceDispatch.destroyImageView));
    }

    const render_pass = try defaultRenderPass(device.handle, swapchain.image_format);
    try deletion_queue.append(VulkanDeleter.make(render_pass, DeviceDispatch.destroyRenderPass));

    const framebuffers = try createFramebuffers(allocator, device.handle, render_pass, swapchain.extent, image_views);
    for (framebuffers) |framebuffer| {
        try deletion_queue.append(VulkanDeleter.make(framebuffer, DeviceDispatch.destroyFramebuffer));
    }

    const command_pool_info = vk.CommandPoolCreateInfo{
        .flags = .{ .reset_command_buffer_bit = true },
        .queue_family_index = device.physical_device.graphics_family_index,
    };
    const command_pool = try vkd().createCommandPool(device.handle, &command_pool_info, null);
    try deletion_queue.append(VulkanDeleter.make(command_pool, DeviceDispatch.destroyCommandPool));

    const command_buffer_info = vk.CommandBufferAllocateInfo{
        .command_pool = command_pool,
        .command_buffer_count = 1,
        .level = .primary,
    };
    var command_buffer: vk.CommandBuffer = .null_handle;
    try vkd().allocateCommandBuffers(device.handle, &command_buffer_info, @ptrCast(&command_buffer));

    const sync = try createSyncObjects(device.handle);
    try deletion_queue.append(VulkanDeleter.make(sync.render_fence, DeviceDispatch.destroyFence));
    try deletion_queue.append(VulkanDeleter.make(sync.render_semaphore, DeviceDispatch.destroySemaphore));
    try deletion_queue.append(VulkanDeleter.make(sync.present_semaphore, DeviceDispatch.destroySemaphore));

    const triangle_shader_vert = try createShaderModule(device.handle, &Shaders.colored_triangle_vert);
    const triangle_shader_frag = try createShaderModule(device.handle, &Shaders.colored_triangle_frag);
    const red_triangle_shader_vert = try createShaderModule(device.handle, &Shaders.triangle_vert);
    const red_triangle_shader_frag = try createShaderModule(device.handle, &Shaders.triangle_frag);
    const triangle_mesh_shader_vert = try createShaderModule(device.handle, &Shaders.triangle_mesh_vert);

    const pipeline_layout_info = vk.PipelineLayoutCreateInfo{};
    const pipeline_layout = try vkd().createPipelineLayout(device.handle, &pipeline_layout_info, null);
    try deletion_queue.append(VulkanDeleter.make(pipeline_layout, DeviceDispatch.destroyPipelineLayout));

    var shader_stages = std.ArrayList(vk.PipelineShaderStageCreateInfo).init(allocator);
    defer shader_stages.deinit();

    try shader_stages.append(vk_init.pipelineShaderStageCreateInfo(.{ .vertex_bit = true }, triangle_shader_vert));
    try shader_stages.append(vk_init.pipelineShaderStageCreateInfo(.{ .fragment_bit = true }, triangle_shader_frag));
    var pipeline_builder = PipelineBuilder{
        .shader_stages = shader_stages,
        .vertex_input_info = vk.PipelineVertexInputStateCreateInfo{},
        .input_assembly = vk_init.inputAssemblyCreateInfo(.triangle_list),
        .viewport = .{
            .x = 0,
            .y = 0,
            .width = @floatFromInt(swapchain.extent.width),
            .height = @floatFromInt(swapchain.extent.height),
            .min_depth = 0,
            .max_depth = 1,
        },
        .scissor = .{
            .offset = .{ .x = 0, .y = 0 },
            .extent = swapchain.extent,
        },
        .rasterizer = vk_init.rasterizationStateCreateInfo(.fill),
        .multisampling = vk_init.multisamplingStateCreateInfo(),
        .color_blend_attachment = vk_init.colorBlendAttachmentState(),
        .pipeline_layout = pipeline_layout,
    };

    const pipeline = pipeline_builder.buildPipeline(device.handle, render_pass);
    if (pipeline == null) return error.PipelineCreationFailed;
    try deletion_queue.append(VulkanDeleter.make(pipeline.?, DeviceDispatch.destroyPipeline));

    shader_stages.clearRetainingCapacity();
    try shader_stages.append(vk_init.pipelineShaderStageCreateInfo(.{ .vertex_bit = true }, red_triangle_shader_vert));
    try shader_stages.append(vk_init.pipelineShaderStageCreateInfo(.{ .fragment_bit = true }, red_triangle_shader_frag));
    pipeline_builder.shader_stages = shader_stages;

    const red_pipeline = pipeline_builder.buildPipeline(device.handle, render_pass);
    if (red_pipeline == null) return error.PipelineCreationFailed;
    try deletion_queue.append(VulkanDeleter.make(red_pipeline.?, DeviceDispatch.destroyPipeline));

    const push_constant = vk.PushConstantRange{
        .offset = 0,
        .size = @sizeOf(MeshPushConstants),
        .stage_flags = .{ .vertex_bit = true },
    };
    const mesh_pipeline_layout_info = vk.PipelineLayoutCreateInfo{
        .push_constant_range_count = 1,
        .p_push_constant_ranges = @ptrCast(&push_constant),
    };

    const mesh_pipeline_layout = try vkd().createPipelineLayout(device.handle, &mesh_pipeline_layout_info, null);
    try deletion_queue.append(VulkanDeleter.make(mesh_pipeline_layout, DeviceDispatch.destroyPipelineLayout));

    pipeline_builder.pipeline_layout = mesh_pipeline_layout;

    shader_stages.clearRetainingCapacity();
    try shader_stages.append(vk_init.pipelineShaderStageCreateInfo(.{ .vertex_bit = true }, triangle_mesh_shader_vert));
    try shader_stages.append(vk_init.pipelineShaderStageCreateInfo(.{ .fragment_bit = true }, triangle_shader_frag));
    pipeline_builder.shader_stages = shader_stages;

    const vertex_description = try Mesh.Vertex.getVertexDescription(allocator);
    defer {
        vertex_description.bindings.deinit();
        vertex_description.attributes.deinit();
    }
    pipeline_builder.vertex_input_info.vertex_binding_description_count = @intCast(vertex_description.bindings.items.len);
    pipeline_builder.vertex_input_info.p_vertex_binding_descriptions = vertex_description.bindings.items.ptr;
    pipeline_builder.vertex_input_info.vertex_attribute_description_count = @intCast(vertex_description.attributes.items.len);
    pipeline_builder.vertex_input_info.p_vertex_attribute_descriptions = vertex_description.attributes.items.ptr;

    const mesh_pipeline = pipeline_builder.buildPipeline(device.handle, render_pass);
    if (mesh_pipeline == null) return error.PipelineCreationFailed;
    try deletion_queue.append(VulkanDeleter.make(mesh_pipeline.?, DeviceDispatch.destroyPipeline));

    vkd().destroyShaderModule(device.handle, triangle_shader_vert, null);
    vkd().destroyShaderModule(device.handle, triangle_shader_frag, null);
    vkd().destroyShaderModule(device.handle, red_triangle_shader_vert, null);
    vkd().destroyShaderModule(device.handle, red_triangle_shader_frag, null);
    vkd().destroyShaderModule(device.handle, triangle_mesh_shader_vert, null);

    var buffer_deletion_queue = std.ArrayList(AllocatedBuffer).init(allocator);

    const mesh = try loadMesh(allocator, vma, &buffer_deletion_queue);

    return .{
        .allocator = allocator,
        .vma = vma,
        .window = window,
        .instance = instance,
        .device = device,
        .surface = surface,
        .deletion_queue = deletion_queue,
        .buffer_deletion_queue = buffer_deletion_queue,
        .swapchain = swapchain,
        .swapchain_images = images,
        .swapchain_image_views = image_views,
        .command_pool = command_pool,
        .main_command_buffer = command_buffer,
        .render_pass = render_pass,
        .framebuffers = framebuffers,
        .render_fence = sync.render_fence,
        .present_semaphore = sync.present_semaphore,
        .render_semaphore = sync.render_semaphore,
        .triangle_pipeline_layout = pipeline_layout,
        .triangle_pipeline = pipeline.?,
        .red_triangle_pipeline = red_pipeline.?,
        .mesh_pipeline_layout = mesh_pipeline_layout,
        .mesh_pipeline = mesh_pipeline.?,
        .triangle_mesh = mesh,
    };
}

pub fn deinit(self: *@This()) void {
    self.triangle_mesh.vertices.deinit();
    flushBufferDeletionQueue(self.vma, self.buffer_deletion_queue.items);
    self.buffer_deletion_queue.deinit();
    c.vmaDestroyAllocator(self.vma);
    flushDeletionQueue(self.device.handle, self.deletion_queue.items);
    self.deletion_queue.deinit();
    self.allocator.free(self.framebuffers);
    self.allocator.free(self.swapchain_image_views);
    self.allocator.free(self.swapchain_images);
    self.device.destroy();
    vki().destroySurfaceKHR(self.instance.handle, self.surface, null);
    self.instance.destroy();
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

        if (self.window.keyPressed(c.GLFW_KEY_SPACE)) {
            self.selected_shader ^= 1;
        }

        if (self.stop_rendering) {
            std.time.sleep(std.time.ns_per_ms * 100);
            continue;
        }

        try self.draw();
    }
}

pub fn waitForIdle(self: *const @This()) !void {
    try vkd().deviceWaitIdle(self.device.handle);
}

fn loadMesh(
    allocator: Allocator,
    vma: c.VmaAllocator,
    deletion_queue: *std.ArrayList(AllocatedBuffer),
) !Mesh {
    var vertices = try std.ArrayList(Mesh.Vertex).initCapacity(allocator, 3);
    errdefer vertices.deinit();

    try vertices.append(.{
        .position = .{ 1, 1, 0 },
        .normal = .{ 0, 0, 0 },
        .color = .{ 0, 1, 0 },
    });
    try vertices.append(.{
        .position = .{ -1, 1, 0 },
        .normal = .{ 0, 0, 0 },
        .color = .{ 0, 1, 0 },
    });
    try vertices.append(.{
        .position = .{ 0, -1, 0 },
        .normal = .{ 0, 0, 0 },
        .color = .{ 0, 1, 0 },
    });

    const buffer = try createMeshBuffer(vma, vertices.items.len * @sizeOf(Mesh.Vertex), deletion_queue);

    var data: ?*anyopaque = null;
    try vkCheck(c.vmaMapMemory(vma, buffer.allocation, &data));

    const ptr: [*]Mesh.Vertex = @ptrCast(@alignCast(data));
    @memcpy(ptr, vertices.items);

    c.vmaUnmapMemory(vma, buffer.allocation);

    return .{
        .vertices = vertices,
        .vertex_buffer = buffer,
    };
}

fn createMeshBuffer(
    vma: c.VmaAllocator,
    size: vk.DeviceSize,
    deletion_queue: *std.ArrayList(AllocatedBuffer),
) !AllocatedBuffer {
    const buffer_info = c.VkBufferCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
    };

    const vma_alloc_info = c.VmaAllocationCreateInfo{
        .usage = c.VMA_MEMORY_USAGE_CPU_TO_GPU,
    };
    var buffer: c.VkBuffer = undefined;
    var allocation: c.VmaAllocation = undefined;
    try vkCheck(c.vmaCreateBuffer(vma, &buffer_info, &vma_alloc_info, &buffer, &allocation, null));

    const allocated_buffer = AllocatedBuffer{
        .buffer = c.vulkanCHandleToZig(vk.Buffer, buffer),
        .allocation = allocation,
    };
    try deletion_queue.append(allocated_buffer);

    return allocated_buffer;
}

fn uploadMesh(self: *const @This(), mesh: *Mesh) !void {
    _ = self;
    _ = mesh;
}

const VulkanDeleter = struct {
    handle: usize,
    delete_fn: *const fn (self: *const @This(), device: vk.Device) void,

    fn make(handle: anytype, func: anytype) @This() {
        const T = @TypeOf(handle);
        const info = @typeInfo(T);
        if (info != .Enum) @compileError("handle must be a Vulkan handle");

        const Fn = @TypeOf(func);
        if (@typeInfo(Fn) != .Fn) @compileError("func must be a function");

        const Deleter = struct {
            fn delete_impl(deleter: *const VulkanDeleter, device: vk.Device) void {
                const h: T = @enumFromInt(deleter.handle);
                func(vkd(), device, h, null);
            }
        };

        return .{
            .handle = @intFromEnum(handle),
            .delete_fn = Deleter.delete_impl,
        };
    }

    fn delete(self: *const @This(), device: vk.Device) void {
        self.delete_fn(self, device);
    }
};

fn flushDeletionQueue(device: vk.Device, entries: []const VulkanDeleter) void {
    var it = std.mem.reverseIterator(entries);
    while (it.next()) |entry| {
        entry.delete(device);
    }
}

fn flushBufferDeletionQueue(vma: c.VmaAllocator, entries: []const AllocatedBuffer) void {
    var it = std.mem.reverseIterator(entries);
    while (it.next()) |entry| {
        c.vmaDestroyBuffer(vma, c.vulkanZigHandleToC(c.VkBuffer, entry.buffer), entry.allocation);
    }
}

fn draw(self: *@This()) !void {
    var result = try vkd().waitForFences(self.device.handle, 1, @ptrCast(&self.render_fence), vk.TRUE, std.time.ns_per_s);
    std.debug.assert(result == .success);
    try vkd().resetFences(self.device.handle, 1, @ptrCast(&self.render_fence));

    const next_image_result = try vkd().acquireNextImageKHR(
        self.device.handle,
        self.swapchain.handle,
        std.time.ns_per_s,
        self.present_semaphore,
        .null_handle,
    );
    std.debug.assert(next_image_result.result == .success);

    const image_index = next_image_result.image_index;

    try vkd().resetCommandBuffer(self.main_command_buffer, .{});

    const cmd = self.main_command_buffer;

    const cmd_begin_info = vk.CommandBufferBeginInfo{
        .flags = .{ .one_time_submit_bit = true },
    };
    try vkd().beginCommandBuffer(cmd, &cmd_begin_info);

    const flash = @abs(@sin(@as(f32, @floatFromInt(self.frame_number)) / 120));
    const clear_value = vk.ClearValue{ .color = .{ .float_32 = .{ 0, 0, flash, 1 } } };

    const render_pass_info = vk.RenderPassBeginInfo{
        .render_pass = self.render_pass,
        .framebuffer = self.framebuffers[image_index],
        .render_area = .{
            .offset = .{ .x = 0, .y = 0 },
            .extent = self.swapchain.extent,
        },
        .clear_value_count = 1,
        .p_clear_values = @ptrCast(&clear_value),
    };
    vkd().cmdBeginRenderPass(cmd, &render_pass_info, .@"inline");

    vkd().cmdBindVertexBuffers(cmd, 0, 1, @ptrCast(&self.triangle_mesh.vertex_buffer.buffer), &[_]vk.DeviceSize{0});

    vkd().cmdBindPipeline(cmd, .graphics, self.mesh_pipeline);

    const view = math.mat.lookAt(.{ 0, 0, 2 }, .{ 0, 0, 0 }, .{ 0, -1, 0 });
    const projection = math.mat.perspective(std.math.degreesToRadians(f32, 70), self.window.aspectRatio(), 0.1, 200);
    const model = math.mat.rotation(std.math.degreesToRadians(f32, @as(f32, @floatFromInt(self.frame_number)) * 0.4), .{ 0, 1, 0 });
    const mesh_matrix = math.mat.mul(&projection, &math.mat.mul(&view, &model));

    const push = MeshPushConstants{
        .data = .{ 0, 0, 0, 0 },
        .render_matrix = mesh_matrix,
    };

    vkd().cmdPushConstants(cmd, self.mesh_pipeline_layout, .{ .vertex_bit = true }, 0, @sizeOf(MeshPushConstants), @ptrCast(&push));

    vkd().cmdDraw(cmd, @intCast(self.triangle_mesh.vertices.items.len), 1, 0, 0);

    vkd().cmdEndRenderPass(cmd);
    try vkd().endCommandBuffer(cmd);

    const wait_stage = vk.PipelineStageFlags{ .color_attachment_output_bit = true };
    const submit = vk.SubmitInfo{
        .p_wait_dst_stage_mask = @ptrCast(&wait_stage),
        .wait_semaphore_count = 1,
        .p_wait_semaphores = @ptrCast(&self.present_semaphore),
        .signal_semaphore_count = 1,
        .p_signal_semaphores = @ptrCast(&self.render_semaphore),
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast(&cmd),
    };
    try vkd().queueSubmit(self.device.graphics_queue, 1, @ptrCast(&submit), self.render_fence);

    const present_info = vk.PresentInfoKHR{
        .swapchain_count = 1,
        .p_swapchains = @ptrCast(&self.swapchain.handle),
        .wait_semaphore_count = 1,
        .p_wait_semaphores = @ptrCast(&self.render_semaphore),
        .p_image_indices = @ptrCast(&image_index),
    };
    result = try vkd().queuePresentKHR(self.device.graphics_queue, &present_info);
    std.debug.assert(result == .success);

    self.frame_number += 1;
}

fn createShaderModule(device: vk.Device, bytecode: []align(4) const u8) !vk.ShaderModule {
    const create_info = vk.ShaderModuleCreateInfo{
        .code_size = bytecode.len,
        .p_code = std.mem.bytesAsSlice(u32, bytecode).ptr,
    };

    return vkd().createShaderModule(device, &create_info, null);
}

const SyncObjects = struct {
    render_fence: vk.Fence,
    present_semaphore: vk.Semaphore,
    render_semaphore: vk.Semaphore,
};

fn destroySyncObjects(device: vk.Device, sync: SyncObjects) void {
    vkd().destroyFence(device, sync.render_fence, null);
    vkd().destroySemaphore(device, sync.present_semaphore, null);
    vkd().destroySemaphore(device, sync.render_semaphore, null);
}

fn createSyncObjects(device: vk.Device) !SyncObjects {
    const fence_info = vk_init.fenceCreateInfo(.{ .signaled_bit = true });
    const fence = try vkd().createFence(device, &fence_info, null);
    errdefer vkd().destroyFence(device, fence, null);

    const semaphore_info = vk_init.semaphoreCreateInfo(.{});
    const present_semaphore = try vkd().createSemaphore(device, &semaphore_info, null);
    errdefer vkd().destroySemaphore(device, present_semaphore, null);
    const render_semaphore = try vkd().createSemaphore(device, &semaphore_info, null);
    errdefer vkd().destroySemaphore(device, render_semaphore, null);

    return .{
        .render_fence = fence,
        .present_semaphore = present_semaphore,
        .render_semaphore = render_semaphore,
    };
}

fn createFramebuffers(
    allocator: Allocator,
    device: vk.Device,
    render_pass: vk.RenderPass,
    extent: vk.Extent2D,
    image_views: []const vk.ImageView,
) ![]vk.Framebuffer {
    var framebuffer_info = vk.FramebufferCreateInfo{
        .render_pass = render_pass,
        .attachment_count = 1,
        .width = extent.width,
        .height = extent.height,
        .layers = 1,
    };

    var framebuffers = try std.ArrayList(vk.Framebuffer).initCapacity(allocator, image_views.len);
    errdefer {
        for (framebuffers.items) |framebuffer| {
            vkd().destroyFramebuffer(device, framebuffer, null);
        }
        framebuffers.deinit();
    }

    for (0..image_views.len) |i| {
        framebuffer_info.p_attachments = @ptrCast(&image_views[i]);
        const framebuffer = try vkd().createFramebuffer(device, &framebuffer_info, null);
        try framebuffers.append(framebuffer);
    }

    return framebuffers.toOwnedSlice();
}

fn defaultRenderPass(device: vk.Device, image_format: vk.Format) !vk.RenderPass {
    const color_attachment = vk.AttachmentDescription{
        .format = image_format,
        .samples = .{ .@"1_bit" = true },
        .load_op = .clear,
        .store_op = .store,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = .undefined,
        .final_layout = .present_src_khr,
    };

    const color_attachment_ref = vk.AttachmentReference{
        .attachment = 0,
        .layout = .color_attachment_optimal,
    };

    const subpass = vk.SubpassDescription{
        .pipeline_bind_point = .graphics,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&color_attachment_ref),
    };

    const render_pass_info = vk.RenderPassCreateInfo{
        .attachment_count = 1,
        .p_attachments = @ptrCast(&color_attachment),
        .subpass_count = 1,
        .p_subpasses = @ptrCast(&subpass),
    };

    return vkd().createRenderPass(device, &render_pass_info, null);
}

const PipelineBuilder = struct {
    shader_stages: std.ArrayList(vk.PipelineShaderStageCreateInfo),
    vertex_input_info: vk.PipelineVertexInputStateCreateInfo,
    input_assembly: vk.PipelineInputAssemblyStateCreateInfo,
    viewport: vk.Viewport,
    scissor: vk.Rect2D,
    rasterizer: vk.PipelineRasterizationStateCreateInfo,
    multisampling: vk.PipelineMultisampleStateCreateInfo,
    color_blend_attachment: vk.PipelineColorBlendAttachmentState,
    pipeline_layout: vk.PipelineLayout,

    fn buildPipeline(self: *const @This(), device: vk.Device, render_pass: vk.RenderPass) ?vk.Pipeline {
        const viewport_state = vk.PipelineViewportStateCreateInfo{
            .viewport_count = 1,
            .p_viewports = @ptrCast(&self.viewport),
            .scissor_count = 1,
            .p_scissors = @ptrCast(&self.scissor),
        };

        const color_blending = vk.PipelineColorBlendStateCreateInfo{
            .logic_op_enable = vk.FALSE,
            .logic_op = .copy,
            .attachment_count = 1,
            .p_attachments = @ptrCast(&self.color_blend_attachment),
            .blend_constants = .{ 0, 0, 0, 0 },
        };

        const pipeline_info = vk.GraphicsPipelineCreateInfo{
            .stage_count = @intCast(self.shader_stages.items.len),
            .p_stages = self.shader_stages.items.ptr,
            .p_vertex_input_state = &self.vertex_input_info,
            .p_input_assembly_state = &self.input_assembly,
            .p_viewport_state = &viewport_state,
            .p_rasterization_state = &self.rasterizer,
            .p_multisample_state = &self.multisampling,
            .p_color_blend_state = &color_blending,
            .layout = self.pipeline_layout,
            .render_pass = render_pass,
            .subpass = 0,
            .base_pipeline_index = -1,
        };

        var graphics_pipeline: vk.Pipeline = .null_handle;
        const result = vkd().createGraphicsPipelines(device, .null_handle, 1, @ptrCast(&pipeline_info), null, @ptrCast(&graphics_pipeline));
        if (result) |res| {
            if (res == .success) {
                return graphics_pipeline;
            } else {
                log.err("failed to create pipeline: {s}", .{@tagName(res)});
                return null;
            }
        } else |err| {
            log.err("failed to create pipeline: {s}", .{@errorName(err)});
            return null;
        }
    }
};

fn vkCheck(result: c.VkResult) !void {
    if (result != c.VK_SUCCESS) return error.VulkanError;
}

fn errorCallback(error_code: i32, description: [*c]const u8) callconv(.C) void {
    const glfw_log = std.log.scoped(.glfw);
    glfw_log.err("{d}: {s}\n", .{ error_code, description });
}
