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
const frame_overlap = 2;

pub const AllocatedBuffer = struct {
    buffer: vk.Buffer,
    allocation: c.VmaAllocation,
};

pub const AllocatedImage = struct {
    image: vk.Image,
    allocation: c.VmaAllocation,
};

pub const MeshPushConstants = extern struct {
    data: math.Vec4 align(16),
    render_matrix: math.Mat4 align(16),
};

pub const GpuCameraData = extern struct {
    view: math.Mat4 align(16),
    proj: math.Mat4 align(16),
    view_proj: math.Mat4 align(16),
};

pub const Material = struct {
    pipeline: vk.Pipeline,
    pipeline_layout: vk.PipelineLayout,
};

pub const RenderObject = struct {
    mesh: *Mesh,
    material: *Material,
    transform_matrix: math.Mat4,
};

pub const FrameData = struct {
    present_semaphore: vk.Semaphore,
    render_semaphore: vk.Semaphore,
    render_fence: vk.Fence,

    command_pool: vk.CommandPool,
    command_buffer: vk.CommandBuffer,

    camera_buffer: AllocatedBuffer,
    global_descriptor: vk.DescriptorSet,
};

const DeletionQueue = std.ArrayList(VulkanDeleter);
const BufferDeletionQueue = std.ArrayList(AllocatedBuffer);
const ImageDeletionQueue = std.ArrayList(AllocatedImage);

frame_number: usize = 0,
stop_rendering: bool = false,

allocator: Allocator,
window: *Window,

vma: c.VmaAllocator,
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
depth_image: AllocatedImage,
depth_image_view: vk.ImageView,
render_pass: vk.RenderPass,

descriptor_pool: vk.DescriptorPool,
descriptor_set_layout: vk.DescriptorSetLayout,

framebuffers: []vk.Framebuffer,
frames: [frame_overlap]FrameData,

renderables: std.ArrayList(RenderObject),
materials: std.StringHashMap(Material),
meshes: std.StringHashMap(Mesh),

pub fn init(allocator: Allocator) !@This() {
    if (c.glfwInit() == c.GLFW_FALSE) return error.GlfwInitFailed;

    _ = c.glfwSetErrorCallback(errorCallback);

    const window = try Window.init(allocator, window_width, window_height, window_title);
    const instance = try vkk.Instance.create(allocator, c.glfwGetInstanceProcAddress, .{});
    const surface = try window.createSurface(instance.handle);
    const physical_device = try vkk.PhysicalDevice.select(allocator, &instance, .{ .surface = surface });
    const device = try vkk.Device.create(allocator, &physical_device, null);
    const vma_info = vk_init.vmaAllocatorCreateInfo(instance.handle, physical_device.handle, device.handle);
    var vma: c.VmaAllocator = undefined;
    try vkCheck(c.vmaCreateAllocator(&vma_info, &vma));

    var deletion_queue = DeletionQueue.init(allocator);
    var buffer_deletion_queue = BufferDeletionQueue.init(allocator);
    var image_deletion_queue = ImageDeletionQueue.init(allocator);

    const swapchain = try vkk.Swapchain.create(allocator, &device, surface, .{
        .desired_extent = window.extent(),
        .desired_present_modes = &.{
            .fifo_khr,
        },
    });
    try deletion_queue.append(VulkanDeleter.make(swapchain.handle, DeviceDispatch.destroySwapchainKHR));

    const swapchain_images = try swapchain.getImages(allocator);
    const swapchain_image_views = try swapchain.getImageViews(allocator, swapchain_images);
    for (swapchain_image_views) |view| {
        try deletion_queue.append(VulkanDeleter.make(view, DeviceDispatch.destroyImageView));
    }

    const depth_format: vk.Format = .d32_sfloat;
    const depth_image = try createDepthImage(vma, depth_format, swapchain.extent);
    try image_deletion_queue.append(depth_image);

    const depth_image_view_info = vk_init.imageViewCreateInfo(depth_format, depth_image.image, .{ .depth_bit = true });
    const depth_image_view = try vkd().createImageView(device.handle, &depth_image_view_info, null);
    try deletion_queue.append(VulkanDeleter.make(depth_image_view, DeviceDispatch.destroyImageView));

    const render_pass = try defaultRenderPass(device.handle, swapchain.image_format, depth_format);
    try deletion_queue.append(VulkanDeleter.make(render_pass, DeviceDispatch.destroyRenderPass));

    const framebuffers = try createFramebuffers(allocator, device.handle, render_pass, swapchain.extent, swapchain_image_views, depth_image_view);
    for (framebuffers) |framebuffer| {
        try deletion_queue.append(VulkanDeleter.make(framebuffer, DeviceDispatch.destroyFramebuffer));
    }

    const descriptor_set_layout = try createDescriptorSetLayout(device.handle);
    try deletion_queue.append(VulkanDeleter.make(descriptor_set_layout, DeviceDispatch.destroyDescriptorSetLayout));

    const descriptor_pool = try createDescriptorPool(device.handle);
    try deletion_queue.append(VulkanDeleter.make(descriptor_pool, DeviceDispatch.destroyDescriptorPool));

    const frames = try createFrameData(vma, device.handle, physical_device.graphics_family_index, descriptor_set_layout, descriptor_pool);
    for (frames) |frame| {
        try deletion_queue.append(VulkanDeleter.make(frame.command_pool, DeviceDispatch.destroyCommandPool));
        try deletion_queue.append(VulkanDeleter.make(frame.render_fence, DeviceDispatch.destroyFence));
        try deletion_queue.append(VulkanDeleter.make(frame.render_semaphore, DeviceDispatch.destroySemaphore));
        try deletion_queue.append(VulkanDeleter.make(frame.present_semaphore, DeviceDispatch.destroySemaphore));
        try buffer_deletion_queue.append(frame.camera_buffer);
    }

    var engine: @This() = .{
        .allocator = allocator,
        .window = window,
        .deletion_queue = deletion_queue,
        .image_deletion_queue = image_deletion_queue,
        .buffer_deletion_queue = buffer_deletion_queue,
        .meshes = std.StringHashMap(Mesh).init(allocator),
        .materials = std.StringHashMap(Material).init(allocator),
        .renderables = std.ArrayList(RenderObject).init(allocator),
        .instance = instance,
        .surface = surface,
        .device = device,
        .vma = vma,
        .swapchain = swapchain,
        .swapchain_images = swapchain_images,
        .swapchain_image_views = swapchain_image_views,
        .depth_format = depth_format,
        .depth_image = depth_image,
        .depth_image_view = depth_image_view,
        .render_pass = render_pass,
        .framebuffers = framebuffers,
        .frames = frames,
        .descriptor_set_layout = descriptor_set_layout,
        .descriptor_pool = descriptor_pool,
    };

    try engine.initPipelines();
    try engine.initMeshes();
    try engine.initScene();

    return engine;
}

pub fn deinit(self: *@This()) void {
    self.renderables.deinit();
    self.materials.deinit();
    var it = self.meshes.iterator();
    while (it.next()) |entry| {
        entry.value_ptr.vertices.deinit();
    }
    self.meshes.deinit();
    flushImageDeletionQueue(self.vma, self.image_deletion_queue.items);
    self.image_deletion_queue.deinit();
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

fn createDescriptorPool(device: vk.Device) !vk.DescriptorPool {
    const pool_size = vk.DescriptorPoolSize{
        .descriptor_count = 10,
        .type = .uniform_buffer,
    };

    const pool_info = vk.DescriptorPoolCreateInfo{
        .max_sets = 10,
        .pool_size_count = 1,
        .p_pool_sizes = @ptrCast(&pool_size),
    };

    return try vkd().createDescriptorPool(device, &pool_info, null);
}

fn createDescriptorSetLayout(device: vk.Device) !vk.DescriptorSetLayout {
    const buffer_binding = vk.DescriptorSetLayoutBinding{
        .binding = 0,
        .descriptor_count = 1,
        .descriptor_type = .uniform_buffer,
        .stage_flags = .{ .vertex_bit = true },
    };

    const set_info = vk.DescriptorSetLayoutCreateInfo{
        .binding_count = 1,
        .p_bindings = @ptrCast(&buffer_binding),
    };

    return try vkd().createDescriptorSetLayout(device, &set_info, null);
}

fn createBuffer(vma: c.VmaAllocator, size: vk.DeviceSize, usage: vk.BufferUsageFlags, memory_usage: c.VmaMemoryUsage) !AllocatedBuffer {
    const buffer_info = vk.BufferCreateInfo{
        .size = size,
        .usage = usage,
        .sharing_mode = .exclusive,
    };

    const alloc_info = c.VmaAllocationCreateInfo{ .usage = memory_usage };

    var buffer: c.VkBuffer = undefined;
    var allocation: c.VmaAllocation = undefined;
    try vkCheck(c.vmaCreateBuffer(vma, @ptrCast(&buffer_info), &alloc_info, &buffer, &allocation, null));

    return .{
        .buffer = c.vulkanCHandleToZig(vk.Buffer, buffer),
        .allocation = allocation,
    };
}

fn initScene(self: *@This()) !void {
    for (0..40) |x| {
        for (0..40) |y| {
            var x_pos: f32 = @floatFromInt(x);
            x_pos -= 20;
            var y_pos: f32 = @floatFromInt(y);
            y_pos -= 20;
            var transform = math.mat.identity(math.Mat4);
            transform = math.mat.translate(&transform, .{ x_pos, 0, y_pos });
            transform = math.mat.scale(&transform, .{ 0.2, 0.2, 0.2 });
            const tri = RenderObject{
                .material = self.materials.getPtr("defaultmesh").?,
                .mesh = self.meshes.getPtr("triangle").?,
                .transform_matrix = transform,
            };
            try self.renderables.append(tri);
        }
    }
}

fn initMeshes(self: *@This()) !void {
    const triangle_vertices = try makeTriangle(self.allocator);
    var mesh = try uploadMesh(self.vma, triangle_vertices, &self.buffer_deletion_queue);
    try self.meshes.put("triangle", mesh);
    const monkey_vertices = try Mesh.loadFromFile(self.allocator, "assets/monkey_smooth.obj");
    mesh = try uploadMesh(self.vma, monkey_vertices, &self.buffer_deletion_queue);
    try self.meshes.put("monkey", mesh);

    const monkey = RenderObject{
        .material = self.materials.getPtr("defaultmesh").?,
        .mesh = self.meshes.getPtr("monkey").?,
        .transform_matrix = math.mat.identity(math.Mat4),
    };
    try self.renderables.append(monkey);
}

fn initPipelines(self: *@This()) !void {
    const triangle_shader_frag = try createShaderModule(self.device.handle, &Shaders.colored_triangle_frag);
    const triangle_mesh_shader_vert = try createShaderModule(self.device.handle, &Shaders.triangle_mesh_vert);
    defer vkd().destroyShaderModule(self.device.handle, triangle_shader_frag, null);
    defer vkd().destroyShaderModule(self.device.handle, triangle_mesh_shader_vert, null);

    var shader_stages = std.ArrayList(vk.PipelineShaderStageCreateInfo).init(self.allocator);
    defer shader_stages.deinit();

    const push_constant = vk.PushConstantRange{
        .offset = 0,
        .size = @sizeOf(MeshPushConstants),
        .stage_flags = .{ .vertex_bit = true },
    };
    const pipeline_layout_info = vk.PipelineLayoutCreateInfo{
        .push_constant_range_count = 1,
        .p_push_constant_ranges = @ptrCast(&push_constant),
        .set_layout_count = 1,
        .p_set_layouts = @ptrCast(&self.descriptor_set_layout),
    };

    const pipeline_layout = try vkd().createPipelineLayout(self.device.handle, &pipeline_layout_info, null);
    try self.deletion_queue.append(VulkanDeleter.make(pipeline_layout, DeviceDispatch.destroyPipelineLayout));

    try shader_stages.append(vk_init.pipelineShaderStageCreateInfo(.{ .vertex_bit = true }, triangle_mesh_shader_vert));
    try shader_stages.append(vk_init.pipelineShaderStageCreateInfo(.{ .fragment_bit = true }, triangle_shader_frag));
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

    try createMaterial(&self.materials, pipeline.?, pipeline_layout, "defaultmesh");
}

fn createFrameData(
    vma: c.VmaAllocator,
    device: vk.Device,
    graphics_family_index: u32,
    descriptor_set_layout: vk.DescriptorSetLayout,
    descriptor_pool: vk.DescriptorPool,
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

        const sync = try createSyncObjects(device);

        const buffer = try createBuffer(vma, @sizeOf(GpuCameraData), .{ .uniform_buffer_bit = true }, c.VMA_MEMORY_USAGE_CPU_TO_GPU);

        const alloc_info = vk.DescriptorSetAllocateInfo{
            .descriptor_pool = descriptor_pool,
            .descriptor_set_count = 1,
            .p_set_layouts = @ptrCast(&descriptor_set_layout),
        };

        var descriptor_set: vk.DescriptorSet = .null_handle;
        try vkd().allocateDescriptorSets(device, &alloc_info, @ptrCast(&descriptor_set));

        const descriptor_buffer_info = vk.DescriptorBufferInfo{
            .buffer = buffer.buffer,
            .offset = 0,
            .range = @sizeOf(GpuCameraData),
        };

        const set_write = vk.WriteDescriptorSet{
            .dst_binding = 0,
            .dst_set = descriptor_set,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = .uniform_buffer,
            .p_buffer_info = @ptrCast(&descriptor_buffer_info),
            .p_image_info = undefined,
            .p_texel_buffer_view = undefined,
        };

        vkd().updateDescriptorSets(device, 1, @ptrCast(&set_write), 0, null);

        ptr.render_semaphore = sync.render_semaphore;
        ptr.present_semaphore = sync.present_semaphore;
        ptr.render_fence = sync.render_fence;
        ptr.camera_buffer = buffer;
        ptr.global_descriptor = descriptor_set;
    }

    return frames;
}

fn createDepthImage(vma: c.VmaAllocator, depth_format: vk.Format, extent: vk.Extent2D) !AllocatedImage {
    const depth_extent = vk.Extent3D{
        .depth = 1,
        .width = extent.width,
        .height = extent.height,
    };

    const depth_image_info = vk_init.imageCreateInfo(depth_format, .{ .depth_stencil_attachment_bit = true }, depth_extent);
    const depth_image_alloc_info = c.VmaAllocationCreateInfo{
        .usage = c.VMA_MEMORY_USAGE_GPU_ONLY,
        .requiredFlags = c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    };

    var d_image: c.VkImage = undefined;
    var d_image_allocation: c.VmaAllocation = undefined;
    try vkCheck(c.vmaCreateImage(vma, @ptrCast(&depth_image_info), &depth_image_alloc_info, &d_image, &d_image_allocation, null));

    const depth_image = AllocatedImage{
        .image = c.vulkanCHandleToZig(vk.Image, d_image),
        .allocation = d_image_allocation,
    };

    return depth_image;
}

fn currentFrame(self: *const @This()) FrameData {
    return self.frames[self.frame_number % frame_overlap];
}

fn drawObjects(self: *@This(), cmd: vk.CommandBuffer, objects: []const RenderObject) !void {
    const camera_pos = math.Vec3{ 0, 6, 10 };
    const view = math.mat.lookAt(camera_pos, .{ 0, 0, 0 }, .{ 0, -1, 0 });
    const projection = math.mat.perspective(std.math.degreesToRadians(f32, 70), self.window.aspectRatio(), 0.1, 200);

    const camera_data = GpuCameraData{
        .proj = projection,
        .view = view,
        .view_proj = math.mat.mul(&projection, &view),
    };

    const current_frame = self.currentFrame();

    var data: ?*anyopaque = null;
    try vkCheck(c.vmaMapMemory(self.vma, current_frame.camera_buffer.allocation, &data));

    const ptr: *GpuCameraData = @ptrCast(@alignCast(data));
    ptr.* = camera_data;

    c.vmaUnmapMemory(self.vma, current_frame.camera_buffer.allocation);

    var last_mesh: ?*Mesh = null;
    var last_material: ?*Material = null;
    for (objects) |object| {
        const bind_material = last_material == null or object.material != last_material.?;
        const bind_mesh = last_mesh == null or object.mesh != last_mesh.?;

        if (bind_material) {
            vkd().cmdBindPipeline(cmd, .graphics, object.material.pipeline);
            vkd().cmdBindDescriptorSets(cmd, .graphics, object.material.pipeline_layout, 0, 1, @ptrCast(&current_frame.global_descriptor), 0, null);
            last_material = object.material;
        }

        const push = MeshPushConstants{
            .data = .{ 0, 0, 0, 0 },
            .render_matrix = object.transform_matrix,
        };

        vkd().cmdPushConstants(cmd, object.material.pipeline_layout, .{ .vertex_bit = true }, 0, @sizeOf(MeshPushConstants), &push);

        if (bind_mesh) {
            vkd().cmdBindVertexBuffers(cmd, 0, 1, @ptrCast(&object.mesh.vertex_buffer.buffer), &[_]vk.DeviceSize{0});
            last_mesh = object.mesh;
        }

        vkd().cmdDraw(cmd, @intCast(object.mesh.vertices.items.len), 1, 0, 0);
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

fn uploadMesh(
    vma: c.VmaAllocator,
    vertices: std.ArrayList(Mesh.Vertex),
    deletion_queue: *BufferDeletionQueue,
) !Mesh {
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
    deletion_queue: *BufferDeletionQueue,
) !AllocatedBuffer {
    const buffer_info = vk.BufferCreateInfo{
        .size = size,
        .usage = .{ .vertex_buffer_bit = true },
        .sharing_mode = .exclusive,
    };

    const vma_alloc_info = c.VmaAllocationCreateInfo{
        .usage = c.VMA_MEMORY_USAGE_CPU_TO_GPU,
    };
    var buffer: c.VkBuffer = undefined;
    var allocation: c.VmaAllocation = undefined;
    try vkCheck(c.vmaCreateBuffer(vma, @ptrCast(&buffer_info), &vma_alloc_info, &buffer, &allocation, null));

    const allocated_buffer = AllocatedBuffer{
        .buffer = c.vulkanCHandleToZig(vk.Buffer, buffer),
        .allocation = allocation,
    };
    try deletion_queue.append(allocated_buffer);

    return allocated_buffer;
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

fn flushImageDeletionQueue(vma: c.VmaAllocator, entries: []const AllocatedImage) void {
    var it = std.mem.reverseIterator(entries);
    while (it.next()) |entry| {
        c.vmaDestroyImage(vma, c.vulkanZigHandleToC(c.VkImage, entry.image), entry.allocation);
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
    depth_image_view: vk.ImageView,
) ![]vk.Framebuffer {
    var framebuffer_info = vk.FramebufferCreateInfo{
        .render_pass = render_pass,
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
        const attachments = [_]vk.ImageView{ image_views[i], depth_image_view };
        framebuffer_info.attachment_count = attachments.len;
        framebuffer_info.p_attachments = &attachments;
        const framebuffer = try vkd().createFramebuffer(device, &framebuffer_info, null);
        try framebuffers.append(framebuffer);
    }

    return framebuffers.toOwnedSlice();
}

fn defaultRenderPass(device: vk.Device, image_format: vk.Format, depth_format: vk.Format) !vk.RenderPass {
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

    const depth_attachment = vk.AttachmentDescription{
        .format = depth_format,
        .samples = .{ .@"1_bit" = true },
        .load_op = .clear,
        .store_op = .store,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = .undefined,
        .final_layout = .depth_stencil_attachment_optimal,
    };

    const depth_attachment_ref = vk.AttachmentReference{
        .attachment = 1,
        .layout = .depth_stencil_attachment_optimal,
    };

    const subpass = vk.SubpassDescription{
        .pipeline_bind_point = .graphics,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&color_attachment_ref),
        .p_depth_stencil_attachment = @ptrCast(&depth_attachment_ref),
    };

    const dependency = vk.SubpassDependency{
        .src_subpass = vk.SUBPASS_EXTERNAL,
        .dst_subpass = 0,
        .src_stage_mask = .{ .color_attachment_output_bit = true },
        .src_access_mask = .{},
        .dst_stage_mask = .{ .color_attachment_output_bit = true },
        .dst_access_mask = .{ .color_attachment_write_bit = true },
    };

    const depth_dependency = vk.SubpassDependency{
        .src_subpass = vk.SUBPASS_EXTERNAL,
        .dst_subpass = 0,
        .src_stage_mask = .{ .early_fragment_tests_bit = true, .late_fragment_tests_bit = true },
        .src_access_mask = .{},
        .dst_stage_mask = .{ .early_fragment_tests_bit = true, .late_fragment_tests_bit = true },
        .dst_access_mask = .{ .depth_stencil_attachment_write_bit = true },
    };

    const attachments = [_]vk.AttachmentDescription{ color_attachment, depth_attachment };
    const dependencies = [_]vk.SubpassDependency{ dependency, depth_dependency };
    const render_pass_info = vk.RenderPassCreateInfo{
        .attachment_count = attachments.len,
        .p_attachments = &attachments,
        .subpass_count = 1,
        .p_subpasses = @ptrCast(&subpass),
        .dependency_count = dependencies.len,
        .p_dependencies = &dependencies,
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

fn vkCheck(result: c.VkResult) !void {
    if (result != c.VK_SUCCESS) return error.VulkanError;
}

fn errorCallback(error_code: i32, description: [*c]const u8) callconv(.C) void {
    const glfw_log = std.log.scoped(.glfw);
    glfw_log.err("{d}: {s}\n", .{ error_code, description });
}
