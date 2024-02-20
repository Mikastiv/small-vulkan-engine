const std = @import("std");
const c = @import("c.zig");
const vk = @import("vulkan-zig");
const vkk = @import("vk-kickstart");
const vk_init = @import("vk_init.zig");
const vma = @import("vma-zig");
const Engine = @import("Engine.zig");
const Allocator = std.mem.Allocator;

const vkd = vkk.dispatch.vkd;

pub const VulkanDeleter = struct {
    handle: usize,
    delete_fn: *const fn (self: *const @This(), device: vk.Device) void,

    pub fn make(handle: anytype, func: anytype) @This() {
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

    pub fn delete(self: *const @This(), device: vk.Device) void {
        self.delete_fn(self, device);
    }
};

pub const SyncObjects = struct {
    render_fence: vk.Fence,
    present_semaphore: vk.Semaphore,
    render_semaphore: vk.Semaphore,
};

pub fn createBuffer(
    vma_allocator: vma.Allocator,
    size: vk.DeviceSize,
    usage: vk.BufferUsageFlags,
    memory_usage: vma.MemoryUsage,
) !vma.AllocatedBuffer {
    const buffer_info = vk.BufferCreateInfo{
        .size = size,
        .usage = usage,
        .sharing_mode = .exclusive,
    };

    const alloc_info = vma.AllocationCreateInfo{ .usage = memory_usage };

    return vma_allocator.createBuffer(&buffer_info, &alloc_info, null);
}

pub fn createDescriptorSet(device: vk.Device, descriptor_pool: vk.DescriptorPool, layouts: []const vk.DescriptorSetLayout) !vk.DescriptorSet {
    const alloc_info = vk.DescriptorSetAllocateInfo{
        .descriptor_pool = descriptor_pool,
        .descriptor_set_count = @intCast(layouts.len),
        .p_set_layouts = layouts.ptr,
    };

    var descriptor_set: vk.DescriptorSet = .null_handle;
    try vkd().allocateDescriptorSets(device, &alloc_info, @ptrCast(&descriptor_set));

    return descriptor_set;
}

pub fn createDescriptorPool(device: vk.Device) !vk.DescriptorPool {
    const pool_sizes = [_]vk.DescriptorPoolSize{
        .{ .descriptor_count = 10, .type = .uniform_buffer },
        .{ .descriptor_count = 10, .type = .uniform_buffer_dynamic },
        .{ .descriptor_count = 10, .type = .storage_buffer },
        .{ .descriptor_count = 10, .type = .combined_image_sampler },
    };

    const pool_info = vk.DescriptorPoolCreateInfo{
        .max_sets = 10,
        .pool_size_count = pool_sizes.len,
        .p_pool_sizes = &pool_sizes,
    };

    return try vkd().createDescriptorPool(device, &pool_info, null);
}

pub fn createDescriptorSetLayout(
    device: vk.Device,
    bindings: []const vk.DescriptorSetLayoutBinding,
) !vk.DescriptorSetLayout {
    const set_info = vk.DescriptorSetLayoutCreateInfo{
        .binding_count = @intCast(bindings.len),
        .p_bindings = bindings.ptr,
    };

    return try vkd().createDescriptorSetLayout(device, &set_info, null);
}

pub fn createDepthImage(vma_allocator: vma.Allocator, depth_format: vk.Format, extent: vk.Extent2D) !vma.AllocatedImage {
    const depth_extent = vk.Extent3D{
        .depth = 1,
        .width = extent.width,
        .height = extent.height,
    };

    const depth_image_info = vk_init.imageCreateInfo(depth_format, .{ .depth_stencil_attachment_bit = true }, depth_extent);
    const depth_image_alloc_info = vma.AllocationCreateInfo{
        .usage = .gpu_only,
        .required_flags = .{ .device_local_bit = true },
    };

    return vma_allocator.createImage(&depth_image_info, &depth_image_alloc_info, null);
}

pub fn createShaderModule(device: vk.Device, bytecode: []align(4) const u8) !vk.ShaderModule {
    const create_info = vk.ShaderModuleCreateInfo{
        .code_size = bytecode.len,
        .p_code = std.mem.bytesAsSlice(u32, bytecode).ptr,
    };

    return vkd().createShaderModule(device, &create_info, null);
}

pub fn destroySyncObjects(device: vk.Device, sync: SyncObjects) void {
    vkd().destroyFence(device, sync.render_fence, null);
    vkd().destroySemaphore(device, sync.present_semaphore, null);
    vkd().destroySemaphore(device, sync.render_semaphore, null);
}

pub fn createSyncObjects(device: vk.Device) !SyncObjects {
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

pub fn createFramebuffers(
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

pub fn defaultRenderPass(device: vk.Device, image_format: vk.Format, depth_format: vk.Format) !vk.RenderPass {
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

fn vkCheck(result: c.VkResult) !void {
    if (result != c.VK_SUCCESS) return error.VulkanError;
}
