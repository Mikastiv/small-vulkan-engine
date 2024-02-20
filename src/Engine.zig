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
const vma = @import("vma-zig");
const texture = @import("texture.zig");
const Camera = @import("Camera.zig");
const vk_utils = @import("vk_utils.zig");

const vki = vkk.dispatch.vki;
const vkd = vkk.dispatch.vkd;
const DeviceDispatch = vkk.dispatch.DeviceDispatch;

const log = std.log.scoped(.engine);

const window_width = 1700;
const window_height = 900;
const window_title = "Vulkan Engine";

const frame_overlap = 2;
const max_objects = 10000;
const camera_sensivity = 0.5;
const move_speed = 10;

const VulkanDeleter = vk_utils.VulkanDeleter;

const MeshPushConstants = extern struct {
    data: math.Vec4 align(16),
    render_matrix: math.Mat4 align(16),
};

const GpuGlobalData = extern struct {
    view: math.Mat4 align(16),
    proj: math.Mat4 align(16),
    view_proj: math.Mat4 align(16),
    fog_color: math.Vec4 align(16),
    fog_distance: math.Vec4 align(16),
    ambient_color: math.Vec4 align(16),
    sunlight_direction: math.Vec4 align(16),
    sunlight_color: math.Vec4 align(16),
};

const Material = struct {
    texture_set: vk.DescriptorSet = .null_handle,
    pipeline: vk.Pipeline,
    pipeline_layout: vk.PipelineLayout,
};

const RenderObject = struct {
    mesh: *Mesh,
    material: *Material,
    transform_matrix: math.Mat4,
};

const FrameData = struct {
    present_semaphore: vk.Semaphore,
    render_semaphore: vk.Semaphore,
    render_fence: vk.Fence,

    command_pool: vk.CommandPool,
    command_buffer: vk.CommandBuffer,
};

const UploadContext = struct {
    fence: vk.Fence,
    command_pool: vk.CommandPool,
    command_buffer: vk.CommandBuffer,
};

const Texture = struct {
    image: vma.AllocatedImage,
    image_view: vk.ImageView,
};

const DeletionQueue = std.ArrayList(VulkanDeleter);
const BufferDeletionQueue = std.ArrayList(vma.AllocatedBuffer);
const ImageDeletionQueue = std.ArrayList(vma.AllocatedImage);

frame_number: usize = 0,
stop_rendering: bool = false,

allocator: Allocator,
window: *Window,

vma_allocator: vma.Allocator,
instance: vkk.Instance,
device: vkk.Device,

deletion_queue: DeletionQueue,
buffer_deletion_queue: BufferDeletionQueue,
image_deletion_queue: ImageDeletionQueue,

surface: vk.SurfaceKHR,
swapchain: vkk.Swapchain,
swapchain_images: []vk.Image,
swapchain_image_views: []vk.ImageView,
depth_format: vk.Format,
depth_image: vma.AllocatedImage,
depth_image_view: vk.ImageView,
render_pass: vk.RenderPass,

descriptor_pool: vk.DescriptorPool,
global_set_layout: vk.DescriptorSetLayout,
single_texture_set_layout: vk.DescriptorSetLayout,
global_descriptor_set: vk.DescriptorSet,

framebuffers: []vk.Framebuffer,
frames: [frame_overlap]FrameData,

renderables: std.ArrayList(RenderObject),
materials: std.StringHashMap(Material),
meshes: std.StringHashMap(Mesh),
loaded_textures: std.StringHashMap(Texture),

global_gpu_data: GpuGlobalData,
global_buffer: vma.AllocatedBuffer,

upload_context: UploadContext,

camera: Camera,

pub fn init(allocator: Allocator) !@This() {
    if (c.glfwInit() == c.GLFW_FALSE) return error.GlfwInitFailed;

    _ = c.glfwSetErrorCallback(errorCallback);

    const window = try Window.init(allocator, window_width, window_height, window_title);
    const instance = try vkk.Instance.create(allocator, c.glfwGetInstanceProcAddress, .{});
    const surface = try window.createSurface(instance.handle);
    const physical_device = try vkk.PhysicalDevice.select(allocator, &instance, .{
        .surface = surface,
        .required_features = .{
            .wide_lines = vk.TRUE,
        },
    });
    const device = try vkk.Device.create(allocator, &physical_device, null, null);
    const vma_info = vma.AllocatorCreateInfo{
        .instance = instance.handle,
        .physical_device = physical_device.handle,
        .device = device.handle,
    };
    const vma_allocator = try vma.Allocator.create(&vma_info);

    var deletion_queue = DeletionQueue.init(allocator);
    var buffer_deletion_queue = BufferDeletionQueue.init(allocator);
    var image_deletion_queue = ImageDeletionQueue.init(allocator);

    const swapchain = try vkk.Swapchain.create(allocator, &device, surface, .{
        .desired_extent = window.extent(),
        .desired_present_modes = &.{.fifo_khr},
    });
    try deletion_queue.append(VulkanDeleter.make(swapchain.handle, DeviceDispatch.destroySwapchainKHR));

    const swapchain_images = try swapchain.getImages(allocator);
    const swapchain_image_views = try swapchain.getImageViews(allocator, swapchain_images);
    for (swapchain_image_views) |view| {
        try deletion_queue.append(VulkanDeleter.make(view, DeviceDispatch.destroyImageView));
    }

    const depth_format: vk.Format = .d32_sfloat;
    const depth_image = try vk_utils.createDepthImage(vma_allocator, depth_format, swapchain.extent);
    try image_deletion_queue.append(depth_image);

    const depth_image_view_info = vk_init.imageViewCreateInfo(depth_format, depth_image.handle, .{ .depth_bit = true });
    const depth_image_view = try vkd().createImageView(device.handle, &depth_image_view_info, null);
    try deletion_queue.append(VulkanDeleter.make(depth_image_view, DeviceDispatch.destroyImageView));

    const render_pass = try vk_utils.defaultRenderPass(device.handle, swapchain.image_format, depth_format);
    try deletion_queue.append(VulkanDeleter.make(render_pass, DeviceDispatch.destroyRenderPass));

    const framebuffers = try vk_utils.createFramebuffers(allocator, device.handle, render_pass, swapchain.extent, swapchain_image_views, depth_image_view);
    for (framebuffers) |framebuffer| {
        try deletion_queue.append(VulkanDeleter.make(framebuffer, DeviceDispatch.destroyFramebuffer));
    }

    const min_alignment = physical_device.properties.limits.min_uniform_buffer_offset_alignment;
    const global_data_size = frame_overlap * alignUniformBuffer(min_alignment, @sizeOf(GpuGlobalData));
    const global_data_buffer = try vk_utils.createBuffer(vma_allocator, global_data_size, .{ .uniform_buffer_bit = true }, .cpu_to_gpu);
    try buffer_deletion_queue.append(global_data_buffer);

    const descriptor_pool = try vk_utils.createDescriptorPool(device.handle);
    try deletion_queue.append(VulkanDeleter.make(descriptor_pool, DeviceDispatch.destroyDescriptorPool));

    const global_binding = vk_init.descriptorSetLayoutBinding(.uniform_buffer_dynamic, .{ .vertex_bit = true, .fragment_bit = true }, 0);

    const global_set_layout = try vk_utils.createDescriptorSetLayout(device.handle, &.{global_binding});
    try deletion_queue.append(VulkanDeleter.make(global_set_layout, DeviceDispatch.destroyDescriptorSetLayout));

    const global_descriptor_set = try vk_utils.createDescriptorSet(device.handle, descriptor_pool, &.{global_set_layout});

    writeGlobalDescriptorSet(device.handle, global_data_buffer.handle, global_descriptor_set);

    const frames = try createFrameData(device.handle, physical_device.graphics_family_index);
    for (frames) |frame| {
        try deletion_queue.append(VulkanDeleter.make(frame.command_pool, DeviceDispatch.destroyCommandPool));
        try deletion_queue.append(VulkanDeleter.make(frame.render_fence, DeviceDispatch.destroyFence));
        try deletion_queue.append(VulkanDeleter.make(frame.render_semaphore, DeviceDispatch.destroySemaphore));
        try deletion_queue.append(VulkanDeleter.make(frame.present_semaphore, DeviceDispatch.destroySemaphore));
    }

    const upload_context = try createUploadContext(device.handle, device.physical_device.graphics_family_index);
    try deletion_queue.append(VulkanDeleter.make(upload_context.fence, DeviceDispatch.destroyFence));
    try deletion_queue.append(VulkanDeleter.make(upload_context.command_pool, DeviceDispatch.destroyCommandPool));

    const texture_bind = vk_init.descriptorSetLayoutBinding(.combined_image_sampler, .{ .fragment_bit = true }, 0);

    const texture_set_layout = try vk_utils.createDescriptorSetLayout(device.handle, &.{texture_bind});
    try deletion_queue.append(VulkanDeleter.make(texture_set_layout, DeviceDispatch.destroyDescriptorSetLayout));

    var engine: @This() = .{
        .allocator = allocator,
        .window = window,
        .deletion_queue = deletion_queue,
        .image_deletion_queue = image_deletion_queue,
        .buffer_deletion_queue = buffer_deletion_queue,
        .meshes = std.StringHashMap(Mesh).init(allocator),
        .materials = std.StringHashMap(Material).init(allocator),
        .renderables = std.ArrayList(RenderObject).init(allocator),
        .loaded_textures = std.StringHashMap(Texture).init(allocator),
        .instance = instance,
        .surface = surface,
        .device = device,
        .vma_allocator = vma_allocator,
        .swapchain = swapchain,
        .swapchain_images = swapchain_images,
        .swapchain_image_views = swapchain_image_views,
        .depth_format = depth_format,
        .depth_image = depth_image,
        .depth_image_view = depth_image_view,
        .render_pass = render_pass,
        .framebuffers = framebuffers,
        .frames = frames,
        .global_set_layout = global_set_layout,
        .descriptor_pool = descriptor_pool,
        .global_descriptor_set = global_descriptor_set,
        .global_gpu_data = std.mem.zeroes(GpuGlobalData),
        .global_buffer = global_data_buffer,
        .upload_context = upload_context,
        .single_texture_set_layout = texture_set_layout,
        .camera = Camera.init(.{ 0, 6, 10 }),
    };

    try engine.initPipelines();
    try engine.initImages();
    try engine.initMeshes();
    try engine.initScene();

    return engine;
}

pub fn deinit(self: *@This()) void {
    self.renderables.deinit();
    self.materials.deinit();
    {
        var it = self.meshes.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.vertices.deinit();
        }
    }
    self.meshes.deinit();
    self.loaded_textures.deinit();
    flushImageDeletionQueue(self.vma_allocator, self.image_deletion_queue.items);
    self.image_deletion_queue.deinit();
    flushBufferDeletionQueue(self.vma_allocator, self.buffer_deletion_queue.items);
    self.buffer_deletion_queue.deinit();
    self.vma_allocator.destroy();
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
    var timer = try std.time.Timer.start();
    while (!self.window.shouldClose()) {
        const dt_ns = timer.lap();
        const dt: f32 = @as(f32, @floatFromInt(dt_ns)) / std.time.ns_per_s;
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

        try self.update(dt);
        try self.draw();
    }
}

pub fn waitForIdle(self: *const @This()) !void {
    try vkd().deviceWaitIdle(self.device.handle);
}

pub fn immediateSubmit(self: *const @This(), submit_ctx: anytype) !void {
    const cmd = self.upload_context.command_buffer;

    const cmd_begin_info = vk_init.commandBufferBeginInfo(.{ .one_time_submit_bit = true });
    try vkd().beginCommandBuffer(cmd, &cmd_begin_info);

    submit_ctx.recordCommands(cmd);

    try vkd().endCommandBuffer(cmd);

    const submit = vk_init.submitInfo(&.{cmd});
    try vkd().queueSubmit(self.device.graphics_queue, 1, @ptrCast(&submit), self.upload_context.fence);

    const res = try vkd().waitForFences(self.device.handle, 1, @ptrCast(&self.upload_context.fence), vk.TRUE, std.math.maxInt(u64));
    if (res != .success) return error.Timeout;

    try vkd().resetFences(self.device.handle, 1, @ptrCast(&self.upload_context.fence));

    try vkd().resetCommandPool(self.device.handle, self.upload_context.command_pool, .{});
}

fn createUploadContext(device: vk.Device, graphics_family_index: u32) !UploadContext {
    const fence_info = vk_init.fenceCreateInfo(.{});
    const fence = try vkd().createFence(device, &fence_info, null);

    const command_pool_info = vk_init.commandPoolCreateInfo(.{}, graphics_family_index);
    const command_pool = try vkd().createCommandPool(device, &command_pool_info, null);

    const command_buffer_info = vk_init.commandBufferAllocateInfo(command_pool);
    var command_buffer: vk.CommandBuffer = undefined;
    try vkd().allocateCommandBuffers(device, &command_buffer_info, @ptrCast(&command_buffer));

    return .{
        .fence = fence,
        .command_pool = command_pool,
        .command_buffer = command_buffer,
    };
}

fn writeGlobalDescriptorSet(device: vk.Device, buffer: vk.Buffer, set: vk.DescriptorSet) void {
    const global_buffer_info = vk.DescriptorBufferInfo{
        .buffer = buffer,
        .offset = 0,
        .range = @sizeOf(GpuGlobalData),
    };

    const global_write = vk_init.writeDescriptorBuffer(.uniform_buffer_dynamic, set, &global_buffer_info, 0);
    const writes = [_]vk.WriteDescriptorSet{global_write};
    vkd().updateDescriptorSets(device, writes.len, &writes, 0, null);
}

fn alignUniformBuffer(min_ubo_alignment: vk.DeviceSize, size: vk.DeviceSize) vk.DeviceSize {
    if (min_ubo_alignment > 0)
        return std.mem.alignForward(vk.DeviceSize, size, min_ubo_alignment)
    else
        return size;
}

fn update(self: *@This(), dt: f32) !void {
    // for (self.renderables.items) |*object| {
    //     var transform = object.transform_matrix;
    //     transform = math.mat.rotate(&transform, dt, .{ 1, 0, 0 });
    //     object.transform_matrix = transform;
    // }

    const offset = math.vec.mul(self.window.mouseOffset(), camera_sensivity);
    self.camera.update(offset);

    const speed = move_speed * dt;
    const forward = math.vec.mul(self.camera.dir, speed);
    const right = math.vec.mul(self.camera.right, speed);
    if (self.window.key_events[c.GLFW_KEY_W] == c.GLFW_PRESS) self.camera.pos = math.vec.add(self.camera.pos, forward);
    if (self.window.key_events[c.GLFW_KEY_S] == c.GLFW_PRESS) self.camera.pos = math.vec.sub(self.camera.pos, forward);
    if (self.window.key_events[c.GLFW_KEY_A] == c.GLFW_PRESS) self.camera.pos = math.vec.sub(self.camera.pos, right);
    if (self.window.key_events[c.GLFW_KEY_D] == c.GLFW_PRESS) self.camera.pos = math.vec.add(self.camera.pos, right);
}

fn initImages(self: *@This()) !void {
    const lost_empire_image = try texture.loadFromFile(self, "assets/lost_empire-RGBA.png");

    const image_info = vk_init.imageViewCreateInfo(.r8g8b8a8_srgb, lost_empire_image.handle, .{ .color_bit = true });
    const image_view = try vkd().createImageView(self.device.handle, &image_info, null);
    try self.deletion_queue.append(VulkanDeleter.make(image_view, DeviceDispatch.destroyImageView));

    try self.loaded_textures.put("empire_diffuse", .{ .image = lost_empire_image, .image_view = image_view });
}

fn initScene(self: *@This()) !void {
    // for (0..40) |x| {
    //     for (0..40) |y| {
    //         var x_pos: f32 = @floatFromInt(x);
    //         x_pos -= 20;
    //         var y_pos: f32 = @floatFromInt(y);
    //         y_pos -= 20;
    //         var transform = math.mat.identity(math.Mat4);
    //         transform = math.mat.translate(&transform, .{ x_pos, 0, y_pos });
    //         transform = math.mat.scale(&transform, .{ 0.2, 0.2, 0.2 });
    //         const tri = RenderObject{
    //             .material = self.materials.getPtr("defaultmesh").?,
    //             .mesh = self.meshes.getPtr("triangle").?,
    //             .transform_matrix = transform,
    //         };
    //         try self.renderables.append(tri);
    //     }
    // }

    const sampler_info = vk_init.samplerCreateInfo(.nearest, .repeat);
    const sampler = try vkd().createSampler(self.device.handle, &sampler_info, null);
    try self.deletion_queue.append(VulkanDeleter.make(sampler, DeviceDispatch.destroySampler));

    const material = self.materials.getPtr("texturedmesh").?;

    const alloc_info = vk.DescriptorSetAllocateInfo{
        .descriptor_pool = self.descriptor_pool,
        .descriptor_set_count = 1,
        .p_set_layouts = @ptrCast(&self.single_texture_set_layout),
    };

    try vkd().allocateDescriptorSets(self.device.handle, &alloc_info, @ptrCast(&material.texture_set));

    const image_buffer_info = vk.DescriptorImageInfo{
        .sampler = sampler,
        .image_view = self.loaded_textures.getPtr("empire_diffuse").?.image_view,
        .image_layout = .shader_read_only_optimal,
    };

    const texture1 = vk_init.writeDescriptorImage(.combined_image_sampler, material.texture_set, &image_buffer_info, 0);
    vkd().updateDescriptorSets(self.device.handle, 1, @ptrCast(&texture1), 0, null);
}

fn initMeshes(self: *@This()) !void {
    // const triangle_vertices = try makeTriangle(self.allocator);
    // var mesh = try self.uploadMesh(triangle_vertices);
    // try self.meshes.put("triangle", mesh);
    // const monkey_vertices = try Mesh.loadFromFile(self.allocator, "assets/monkey_smooth.obj");
    // mesh = try self.uploadMesh(monkey_vertices);
    // try self.meshes.put("monkey", mesh);

    // const monkey = RenderObject{
    //     .material = self.materials.getPtr("defaultmesh").?,
    //     .mesh = self.meshes.getPtr("monkey").?,
    //     .transform_matrix = math.mat.translation(.{ 0, 3, -3 }),
    // };
    // try self.renderables.append(monkey);

    const lost_empire_vertices = try Mesh.loadFromFile(self.allocator, "assets/lost_empire.obj");
    const empire_mesh = try self.uploadMesh(lost_empire_vertices);
    try self.meshes.put("empire", empire_mesh);

    const map = RenderObject{
        .material = self.materials.getPtr("texturedmesh").?,
        .mesh = self.meshes.getPtr("empire").?,
        .transform_matrix = math.mat.translation(.{ 5, -10, 0 }),
    };
    try self.renderables.append(map);

    const widget_vertices = try makeAxisLines(self.allocator);
    const widget_mesh = try self.uploadMesh(widget_vertices);
    try self.meshes.put("widget", widget_mesh);

    const widget = RenderObject{
        .material = self.materials.getPtr("line").?,
        .mesh = self.meshes.getPtr("widget").?,
        .transform_matrix = math.mat.translation(.{ 0, 3, 0 }),
    };
    try self.renderables.append(widget);

    // const cube_vertices = try Mesh.loadFromFile(self.allocator, "assets/reference.obj");
    // const mesh = try self.uploadMesh(cube_vertices);
    // try self.meshes.put("empire", mesh);

    // const cube = RenderObject{
    //     .material = self.materials.getPtr("texturedmesh").?,
    //     .mesh = self.meshes.getPtr("empire").?,
    //     .transform_matrix = math.mat.translation(.{ 0, 0, -5 }),
    // };
    // try self.renderables.append(cube);
}

fn initPipelines(self: *@This()) !void {
    const triangle_mesh_shader_vert = try vk_utils.createShaderModule(self.device.handle, &Shaders.triangle_mesh_vert);
    defer vkd().destroyShaderModule(self.device.handle, triangle_mesh_shader_vert, null);
    const textured_shader_frag = try vk_utils.createShaderModule(self.device.handle, &Shaders.textured_lit);
    defer vkd().destroyShaderModule(self.device.handle, textured_shader_frag, null);

    const widget_3d_shader_vert = try vk_utils.createShaderModule(self.device.handle, &Shaders.widget_3d_vert);
    defer vkd().destroyShaderModule(self.device.handle, widget_3d_shader_vert, null);
    const widget_3d_shader_frag = try vk_utils.createShaderModule(self.device.handle, &Shaders.widget_3d_frag);
    defer vkd().destroyShaderModule(self.device.handle, widget_3d_shader_frag, null);

    var shader_stages = std.ArrayList(vk.PipelineShaderStageCreateInfo).init(self.allocator);
    defer shader_stages.deinit();

    const push_constant = vk.PushConstantRange{
        .offset = 0,
        .size = @sizeOf(MeshPushConstants),
        .stage_flags = .{ .vertex_bit = true },
    };
    const set_layouts = [_]vk.DescriptorSetLayout{ self.global_set_layout, self.single_texture_set_layout };
    const pipeline_layout_info = vk.PipelineLayoutCreateInfo{
        .push_constant_range_count = 1,
        .p_push_constant_ranges = @ptrCast(&push_constant),
        .set_layout_count = set_layouts.len,
        .p_set_layouts = &set_layouts,
    };

    const pipeline_layout = try vkd().createPipelineLayout(self.device.handle, &pipeline_layout_info, null);
    try self.deletion_queue.append(VulkanDeleter.make(pipeline_layout, DeviceDispatch.destroyPipelineLayout));

    try shader_stages.append(vk_init.pipelineShaderStageCreateInfo(.{ .vertex_bit = true }, triangle_mesh_shader_vert));
    try shader_stages.append(vk_init.pipelineShaderStageCreateInfo(.{ .fragment_bit = true }, textured_shader_frag));
    var pipeline_builder = PipelineBuilder{
        .shader_stages = shader_stages,
        .vertex_input_info = vk.PipelineVertexInputStateCreateInfo{},
        .input_assembly = vk_init.inputAssemblyCreateInfo(.triangle_list),
        .viewport = .{
            .x = 0,
            .y = 0,
            .width = @floatFromInt(self.swapchain.extent.width),
            .height = @floatFromInt(self.swapchain.extent.height),
            .min_depth = 0,
            .max_depth = 1,
        },
        .scissor = .{
            .offset = .{ .x = 0, .y = 0 },
            .extent = self.swapchain.extent,
        },
        .rasterizer = vk_init.rasterizationStateCreateInfo(.fill),
        .multisampling = vk_init.multisamplingStateCreateInfo(),
        .color_blend_attachment = vk_init.colorBlendAttachmentState(),
        .pipeline_layout = pipeline_layout,
        .depth_stencil = vk_init.depthStencilCreateInfo(true, true, .less),
    };

    const vertex_description = try Mesh.Vertex.getVertexDescription(self.allocator);
    defer {
        vertex_description.bindings.deinit();
        vertex_description.attributes.deinit();
    }
    pipeline_builder.vertex_input_info.vertex_binding_description_count = @intCast(vertex_description.bindings.items.len);
    pipeline_builder.vertex_input_info.p_vertex_binding_descriptions = vertex_description.bindings.items.ptr;
    pipeline_builder.vertex_input_info.vertex_attribute_description_count = @intCast(vertex_description.attributes.items.len);
    pipeline_builder.vertex_input_info.p_vertex_attribute_descriptions = vertex_description.attributes.items.ptr;

    const pipeline = pipeline_builder.buildPipeline(self.device.handle, self.render_pass);
    if (pipeline == null) return error.PipelineCreationFailed;
    try self.deletion_queue.append(VulkanDeleter.make(pipeline.?, DeviceDispatch.destroyPipeline));

    try createMaterial(&self.materials, pipeline.?, pipeline_layout, "texturedmesh");

    shader_stages.clearRetainingCapacity();
    try shader_stages.append(vk_init.pipelineShaderStageCreateInfo(.{ .vertex_bit = true }, widget_3d_shader_vert));
    try shader_stages.append(vk_init.pipelineShaderStageCreateInfo(.{ .fragment_bit = true }, widget_3d_shader_frag));
    pipeline_builder.shader_stages = shader_stages;

    pipeline_builder.input_assembly = vk_init.inputAssemblyCreateInfo(.line_list);
    pipeline_builder.rasterizer.line_width = 3;

    const line_pipeline = pipeline_builder.buildPipeline(self.device.handle, self.render_pass);
    if (line_pipeline == null) return error.PipelineCreationFailed;
    try self.deletion_queue.append(VulkanDeleter.make(line_pipeline.?, DeviceDispatch.destroyPipeline));

    try createMaterial(&self.materials, line_pipeline.?, pipeline_layout, "line");
}

fn currentFrame(self: *const @This()) FrameData {
    return self.frames[self.frame_number % frame_overlap];
}

fn drawObjects(self: *@This(), cmd: vk.CommandBuffer, objects: []const RenderObject) !void {
    const view = self.camera.viewMatrix();
    // const view = math.mat.lookAt(.{ -3, 3, 6 }, .{ 0, 3, 0 }, .{ 0, 1, 0 });
    const projection = math.mat.perspective(std.math.degreesToRadians(f32, 70), self.window.aspectRatio(), 0.1, 200);

    self.global_gpu_data.proj = projection;
    self.global_gpu_data.view = view;
    self.global_gpu_data.view_proj = math.mat.mul(&projection, &view);
    const framed: f32 = @as(f32, @floatFromInt(self.frame_number)) / 120.0;
    self.global_gpu_data.ambient_color = .{ @sin(framed) + 0.5, 0, @cos(framed), 1 };

    const alignment = alignUniformBuffer(self.device.physical_device.properties.limits.min_uniform_buffer_offset_alignment, @sizeOf(GpuGlobalData));
    const frame_index = self.frame_number % frame_overlap;
    const uniform_offset = alignment * frame_index;
    {
        const data = try self.vma_allocator.mapMemory(self.global_buffer.allocation);
        const ptr: [*]u8 = @ptrCast(@alignCast(data));

        @memcpy(ptr[uniform_offset .. uniform_offset + @sizeOf(GpuGlobalData)], std.mem.asBytes(&self.global_gpu_data));

        self.vma_allocator.unmapMemory(self.global_buffer.allocation);
    }

    var last_mesh: ?*Mesh = null;
    var last_material: ?*Material = null;
    for (objects, 0..) |object, i| {
        const bind_material = last_material == null or object.material != last_material.?;
        const bind_mesh = last_mesh == null or object.mesh != last_mesh.?;

        if (bind_material) {
            vkd().cmdBindPipeline(cmd, .graphics, object.material.pipeline);
            vkd().cmdBindDescriptorSets(cmd, .graphics, object.material.pipeline_layout, 0, 1, @ptrCast(&self.global_descriptor_set), 1, @ptrCast(&uniform_offset));
            if (object.material.texture_set != .null_handle) {
                vkd().cmdBindDescriptorSets(cmd, .graphics, object.material.pipeline_layout, 1, 1, @ptrCast(&object.material.texture_set), 0, null);
            }
            last_material = object.material;
        }

        const push = MeshPushConstants{
            .data = .{ 0, 0, 0, 0 },
            .render_matrix = object.transform_matrix,
        };

        vkd().cmdPushConstants(cmd, object.material.pipeline_layout, .{ .vertex_bit = true }, 0, @sizeOf(MeshPushConstants), &push);

        if (bind_mesh) {
            vkd().cmdBindVertexBuffers(cmd, 0, 1, @ptrCast(&object.mesh.vertex_buffer.handle), &[_]vk.DeviceSize{0});
            last_mesh = object.mesh;
        }

        vkd().cmdDraw(cmd, @intCast(object.mesh.vertices.items.len), 1, 0, @intCast(i));
    }
}

fn createMaterial(materials: *std.StringHashMap(Material), pipeline: vk.Pipeline, layout: vk.PipelineLayout, name: []const u8) !void {
    const material = Material{
        .pipeline = pipeline,
        .pipeline_layout = layout,
    };
    try materials.put(name, material);
}

fn makeTriangle(allocator: Allocator) !std.ArrayList(Mesh.Vertex) {
    var vertices = try std.ArrayList(Mesh.Vertex).initCapacity(allocator, 3);
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
    return vertices;
}

fn makeAxisLines(allocator: Allocator) !std.ArrayList(Mesh.Vertex) {
    var vertices = try std.ArrayList(Mesh.Vertex).initCapacity(allocator, 3);
    try vertices.append(.{ .position = .{ 0, 0, 0 }, .normal = .{ 0, 0, 0 }, .color = .{ 1, 0, 0 }, .uv = .{ 0, 0 } });
    try vertices.append(.{ .position = .{ 1, 0, 0 }, .normal = .{ 0, 0, 0 }, .color = .{ 1, 0, 0 }, .uv = .{ 0, 0 } });
    try vertices.append(.{ .position = .{ 0, 0, 0 }, .normal = .{ 0, 0, 0 }, .color = .{ 0, 1, 0 }, .uv = .{ 0, 0 } });
    try vertices.append(.{ .position = .{ 0, 1, 0 }, .normal = .{ 0, 0, 0 }, .color = .{ 0, 1, 0 }, .uv = .{ 0, 0 } });
    try vertices.append(.{ .position = .{ 0, 0, 0 }, .normal = .{ 0, 0, 0 }, .color = .{ 0, 0, 1 }, .uv = .{ 0, 0 } });
    try vertices.append(.{ .position = .{ 0, 0, 1 }, .normal = .{ 0, 0, 0 }, .color = .{ 0, 0, 1 }, .uv = .{ 0, 0 } });
    return vertices;
}

fn uploadMesh(
    self: *@This(),
    vertices: std.ArrayList(Mesh.Vertex),
) !Mesh {
    const buffer_size = vertices.items.len * @sizeOf(Mesh.Vertex);

    const staging_buffer = try vk_utils.createBuffer(self.vma_allocator, buffer_size, .{ .transfer_src_bit = true }, .cpu_only);
    defer self.vma_allocator.destroyBuffer(staging_buffer.handle, staging_buffer.allocation);

    {
        const data = try self.vma_allocator.mapMemory(staging_buffer.allocation);
        const ptr: [*]Mesh.Vertex = @ptrCast(@alignCast(data));

        @memcpy(ptr, vertices.items);

        self.vma_allocator.unmapMemory(staging_buffer.allocation);
    }

    const buffer = try vk_utils.createBuffer(self.vma_allocator, buffer_size, .{ .vertex_buffer_bit = true, .transfer_dst_bit = true }, .gpu_only);
    try self.buffer_deletion_queue.append(buffer);

    const MeshCopy = struct {
        staging_buffer: vk.Buffer,
        dst_buffer: vk.Buffer,
        size: vk.DeviceSize,

        fn recordCommands(ctx: @This(), cmd: vk.CommandBuffer) void {
            const copy = vk.BufferCopy{ .size = ctx.size, .src_offset = 0, .dst_offset = 0 };
            vkd().cmdCopyBuffer(cmd, ctx.staging_buffer, ctx.dst_buffer, 1, @ptrCast(&copy));
        }
    };

    try self.immediateSubmit(MeshCopy{
        .staging_buffer = staging_buffer.handle,
        .dst_buffer = buffer.handle,
        .size = buffer_size,
    });

    return .{
        .vertices = vertices,
        .vertex_buffer = buffer,
    };
}

fn flushDeletionQueue(device: vk.Device, entries: []const VulkanDeleter) void {
    var it = std.mem.reverseIterator(entries);
    while (it.next()) |entry| {
        entry.delete(device);
    }
}

fn flushBufferDeletionQueue(vma_allocator: vma.Allocator, entries: []const vma.AllocatedBuffer) void {
    var it = std.mem.reverseIterator(entries);
    while (it.next()) |entry| {
        vma_allocator.destroyBuffer(entry.handle, entry.allocation);
    }
}

fn flushImageDeletionQueue(vma_allocator: vma.Allocator, entries: []const vma.AllocatedImage) void {
    var it = std.mem.reverseIterator(entries);
    while (it.next()) |entry| {
        vma_allocator.destroyImage(entry.handle, entry.allocation);
    }
}

fn draw(self: *@This()) !void {
    const frame = self.currentFrame();

    var result = try vkd().waitForFences(self.device.handle, 1, @ptrCast(&frame.render_fence), vk.TRUE, std.time.ns_per_s);
    std.debug.assert(result == .success);
    try vkd().resetFences(self.device.handle, 1, @ptrCast(&frame.render_fence));

    const next_image_result = try vkd().acquireNextImageKHR(
        self.device.handle,
        self.swapchain.handle,
        std.time.ns_per_s,
        frame.present_semaphore,
        .null_handle,
    );
    std.debug.assert(next_image_result.result == .success);

    const image_index = next_image_result.image_index;

    try vkd().resetCommandBuffer(frame.command_buffer, .{});

    const cmd = frame.command_buffer;

    const cmd_begin_info = vk.CommandBufferBeginInfo{
        .flags = .{ .one_time_submit_bit = true },
    };
    try vkd().beginCommandBuffer(cmd, &cmd_begin_info);

    const flash = @abs(@sin(@as(f32, @floatFromInt(self.frame_number)) / 120));
    const clear_value = vk.ClearValue{ .color = .{ .float_32 = .{ 0, 0, flash, 1 } } };
    const depth_clear = vk.ClearValue{ .depth_stencil = .{ .depth = 1, .stencil = 0 } };

    const clear_values = [_]vk.ClearValue{ clear_value, depth_clear };
    const render_pass_info = vk.RenderPassBeginInfo{
        .render_pass = self.render_pass,
        .framebuffer = self.framebuffers[image_index],
        .render_area = .{
            .offset = .{ .x = 0, .y = 0 },
            .extent = self.swapchain.extent,
        },
        .clear_value_count = clear_values.len,
        .p_clear_values = &clear_values,
    };
    vkd().cmdBeginRenderPass(cmd, &render_pass_info, .@"inline");

    try self.drawObjects(cmd, self.renderables.items);

    vkd().cmdEndRenderPass(cmd);
    try vkd().endCommandBuffer(cmd);

    const wait_stage = vk.PipelineStageFlags{ .color_attachment_output_bit = true };
    const submit = vk.SubmitInfo{
        .p_wait_dst_stage_mask = @ptrCast(&wait_stage),
        .wait_semaphore_count = 1,
        .p_wait_semaphores = @ptrCast(&frame.present_semaphore),
        .signal_semaphore_count = 1,
        .p_signal_semaphores = @ptrCast(&frame.render_semaphore),
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast(&cmd),
    };
    try vkd().queueSubmit(self.device.graphics_queue, 1, @ptrCast(&submit), frame.render_fence);

    const present_info = vk.PresentInfoKHR{
        .swapchain_count = 1,
        .p_swapchains = @ptrCast(&self.swapchain.handle),
        .wait_semaphore_count = 1,
        .p_wait_semaphores = @ptrCast(&frame.render_semaphore),
        .p_image_indices = @ptrCast(&image_index),
    };
    result = try vkd().queuePresentKHR(self.device.graphics_queue, &present_info);
    std.debug.assert(result == .success);

    self.frame_number += 1;
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
    depth_stencil: vk.PipelineDepthStencilStateCreateInfo,

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
            .p_depth_stencil_state = &self.depth_stencil,
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

fn createFrameData(
    device: vk.Device,
    graphics_family_index: u32,
) ![frame_overlap]FrameData {
    const command_pool_info = vk_init.commandPoolCreateInfo(
        .{ .reset_command_buffer_bit = true },
        graphics_family_index,
    );

    var frames: [frame_overlap]FrameData = undefined;
    for (&frames) |*ptr| {
        ptr.command_pool = try vkd().createCommandPool(device, &command_pool_info, null);

        const command_buffer_info = vk_init.commandBufferAllocateInfo(ptr.command_pool);
        try vkd().allocateCommandBuffers(device, &command_buffer_info, @ptrCast(&ptr.command_buffer));

        const sync = try vk_utils.createSyncObjects(device);

        ptr.render_semaphore = sync.render_semaphore;
        ptr.present_semaphore = sync.present_semaphore;
        ptr.render_fence = sync.render_fence;
    }

    return frames;
}

fn errorCallback(error_code: i32, description: [*c]const u8) callconv(.C) void {
    const glfw_log = std.log.scoped(.glfw);
    glfw_log.err("{d}: {s}\n", .{ error_code, description });
}
