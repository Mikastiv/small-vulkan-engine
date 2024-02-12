const std = @import("std");
const c = @import("c.zig");
const Window = @import("Window.zig");
const Allocator = std.mem.Allocator;
const vk = @import("vulkan-zig");
const vkk = @import("vk-kickstart");
const Shaders = @import("shaders");
const vk_init = @import("vk_init.zig");

const vki = vkk.vki;
const vkd = vkk.vkd;

const log = std.log.scoped(.engine);

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
triangle_shader_vert: vk.ShaderModule,
triangle_shader_frag: vk.ShaderModule,
triangle_pipeline_layout: vk.PipelineLayout,
triangle_pipeline: vk.Pipeline,

pub fn init(allocator: Allocator) !@This() {
    if (c.glfwInit() == c.GLFW_FALSE) return error.GlfwInitFailed;
    errdefer c.glfwTerminate();

    _ = c.glfwSetErrorCallback(errorCallback);

    const window = try Window.init(allocator, window_width, window_height, window_title);
    errdefer window.deinit(allocator);

    const instance = try vkk.Instance.create(allocator, c.glfwGetInstanceProcAddress, .{});
    errdefer instance.destroy();

    const surface = try window.createSurface(instance.handle);
    errdefer vki().destroySurfaceKHR(instance.handle, surface, instance.allocation_callbacks);

    const physical_device = try vkk.PhysicalDevice.select(allocator, &instance, .{
        .surface = surface,
    });

    const device = try vkk.Device.create(allocator, &physical_device, null);
    errdefer device.destroy();

    const swapchain = try vkk.Swapchain.create(allocator, &device, surface, .{
        .desired_extent = window.extent(),
        .desired_present_modes = &.{
            .fifo_khr,
        },
    });
    errdefer swapchain.destroy();

    const images = try swapchain.getImages(allocator);
    errdefer allocator.free(images);

    const image_views = try swapchain.getImageViews(allocator, images);
    errdefer swapchain.destroyAndFreeImageViews(allocator, image_views);

    const command_pool_info = vk.CommandPoolCreateInfo{
        .flags = .{ .reset_command_buffer_bit = true },
        .queue_family_index = device.physical_device.graphics_family_index,
    };
    const command_pool = try vkd().createCommandPool(device.handle, &command_pool_info, null);
    errdefer vkd().destroyCommandPool(device.handle, command_pool, null);

    const command_buffer_info = vk.CommandBufferAllocateInfo{
        .command_pool = command_pool,
        .command_buffer_count = 1,
        .level = .primary,
    };
    var command_buffer: vk.CommandBuffer = .null_handle;
    try vkd().allocateCommandBuffers(device.handle, &command_buffer_info, @ptrCast(&command_buffer));

    const render_pass = try defaultRenderPass(device.handle, swapchain.image_format);
    errdefer vkd().destroyRenderPass(device.handle, render_pass, null);

    const framebuffers = try createFramebuffers(allocator, device.handle, render_pass, swapchain.extent, image_views);
    errdefer {
        for (framebuffers) |framebuffer| {
            vkd().destroyFramebuffer(device.handle, framebuffer, null);
        }
        allocator.free(framebuffers);
    }

    const sync = try createSyncObjects(device.handle);
    errdefer destroySyncObjects(device.handle, sync);

    const triangle_shader_vert = try createShaderModule(device.handle, &Shaders.vertex);
    errdefer vkd().destroyShaderModule(device.handle, triangle_shader_vert, null);
    const triangle_shader_frag = try createShaderModule(device.handle, &Shaders.fragment);
    errdefer vkd().destroyShaderModule(device.handle, triangle_shader_frag, null);

    const pipeline_layout_info = vk.PipelineLayoutCreateInfo{};
    const pipeline_layout = try vkd().createPipelineLayout(device.handle, &pipeline_layout_info, null);
    errdefer vkd().destroyPipelineLayout(device.handle, pipeline_layout, null);

    var shader_stages = std.ArrayList(vk.PipelineShaderStageCreateInfo).init(allocator);
    defer shader_stages.deinit();

    try shader_stages.append(.{
        .stage = .{ .vertex_bit = true },
        .module = triangle_shader_vert,
        .p_name = "main",
    });
    try shader_stages.append(.{
        .stage = .{ .fragment_bit = true },
        .module = triangle_shader_frag,
        .p_name = "main",
    });
    const pipeline_builder = PipelineBuilder{
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
    errdefer vkd().destroyPipeline(device.handle, pipeline.?, null);

    return .{
        .allocator = allocator,
        .window = window,
        .instance = instance,
        .device = device,
        .surface = surface,
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
        .triangle_shader_vert = triangle_shader_vert,
        .triangle_shader_frag = triangle_shader_frag,
        .triangle_pipeline_layout = pipeline_layout,
        .triangle_pipeline = pipeline.?,
    };
}

pub fn deinit(self: *@This()) void {
    vkd().destroyPipeline(self.device.handle, self.triangle_pipeline, null);
    vkd().destroyPipelineLayout(self.device.handle, self.triangle_pipeline_layout, null);
    vkd().destroyShaderModule(self.device.handle, self.triangle_shader_vert, null);
    vkd().destroyShaderModule(self.device.handle, self.triangle_shader_frag, null);
    destroySyncObjects(self.device.handle, .{
        .render_fence = self.render_fence,
        .present_semaphore = self.present_semaphore,
        .render_semaphore = self.render_semaphore,
    });
    for (self.framebuffers) |framebuffer| {
        vkd().destroyFramebuffer(self.device.handle, framebuffer, null);
    }
    self.allocator.free(self.framebuffers);
    vkd().destroyRenderPass(self.device.handle, self.render_pass, null);
    vkd().destroyCommandPool(self.device.handle, self.command_pool, null);
    self.swapchain.destroyAndFreeImageViews(self.allocator, self.swapchain_image_views);
    self.allocator.free(self.swapchain_images);
    self.swapchain.destroy();
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

    vkd().cmdBindPipeline(cmd, .graphics, self.triangle_pipeline);
    vkd().cmdDraw(cmd, 3, 1, 0, 0);

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
    const fence_info = vk.FenceCreateInfo{
        .flags = .{ .signaled_bit = true },
    };
    const fence = try vkd().createFence(device, &fence_info, null);
    errdefer vkd().destroyFence(device, fence, null);

    const semaphore_info = vk.SemaphoreCreateInfo{};
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

fn errorCallback(error_code: i32, description: [*c]const u8) callconv(.C) void {
    const glfw_log = std.log.scoped(.glfw);
    glfw_log.err("{d}: {s}\n", .{ error_code, description });
}
