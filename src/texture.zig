const std = @import("std");
const vk = @import("vulkan-zig");
const vkk = @import("vk-kickstart");
const c = @import("c.zig");
const Engine = @import("Engine.zig");
const vk_init = @import("vk_init.zig");
const vk_utils = @import("vk_utils.zig");

const vkd = vkk.dispatch.vkd;

pub fn loadFromFile(engine: *Engine, filename: [*:0]const u8) !Engine.Image {
    var tex_width: c_int = undefined;
    var tex_height: c_int = undefined;
    var tex_channels: c_int = undefined;

    const pixels = c.stbi_load(filename, &tex_width, &tex_height, &tex_channels, c.STBI_rgb_alpha) orelse
        return error.TextureLoadFailed;
    defer c.stbi_image_free(pixels);

    const image_size: vk.DeviceSize = @intCast(tex_width * tex_height * 4);
    const image_format = vk.Format.r8g8b8a8_srgb;

    const staging_buffer = try Engine.createBuffer(&engine.device, image_size, .{ .transfer_src_bit = true }, .{ .host_visible_bit = true });
    defer {
        vkd().destroyBuffer(engine.device.handle, staging_buffer.handle, null);
        vkd().freeMemory(engine.device.handle, staging_buffer.memory, null);
    }

    {
        const data = try vkd().mapMemory(engine.device.handle, staging_buffer.memory, 0, image_size, .{});
        const ptr: [*]c.stbi_uc = @ptrCast(@alignCast(data));

        @memcpy(ptr, pixels[0..image_size]);

        vkd().unmapMemory(engine.device.handle, staging_buffer.memory);
    }

    const extent = vk.Extent3D{
        .width = @intCast(tex_width),
        .height = @intCast(tex_height),
        .depth = 1,
    };
    const image_info = vk_init.imageCreateInfo(image_format, .{ .sampled_bit = true, .transfer_dst_bit = true }, extent);

    const image = try Engine.createImage(&engine.device, &image_info, .{ .device_local_bit = true });

    const ImageCopy = struct {
        image: vk.Image,
        buffer: vk.Buffer,
        queue_family_index: u32,
        extent: vk.Extent3D,

        pub fn recordCommands(ctx: @This(), cmd: vk.CommandBuffer) void {
            const range = vk.ImageSubresourceRange{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            };

            {
                const image_barrier = vk.ImageMemoryBarrier{
                    .old_layout = .undefined,
                    .new_layout = .transfer_dst_optimal,
                    .image = ctx.image,
                    .subresource_range = range,
                    .src_access_mask = .{},
                    .dst_access_mask = .{ .transfer_write_bit = true },
                    .src_queue_family_index = ctx.queue_family_index,
                    .dst_queue_family_index = ctx.queue_family_index,
                };

                vkd().cmdPipelineBarrier(cmd, .{ .top_of_pipe_bit = true }, .{ .transfer_bit = true }, .{}, 0, null, 0, null, 1, @ptrCast(&image_barrier));
            }

            const copy = vk.BufferImageCopy{
                .buffer_offset = 0,
                .buffer_row_length = 0,
                .buffer_image_height = 0,
                .image_subresource = .{
                    .aspect_mask = .{ .color_bit = true },
                    .mip_level = 0,
                    .base_array_layer = 0,
                    .layer_count = 1,
                },
                .image_offset = .{ .x = 0, .y = 0, .z = 0 },
                .image_extent = ctx.extent,
            };

            vkd().cmdCopyBufferToImage(cmd, ctx.buffer, ctx.image, .transfer_dst_optimal, 1, @ptrCast(&copy));

            const image_barrier = vk.ImageMemoryBarrier{
                .old_layout = .transfer_dst_optimal,
                .new_layout = .shader_read_only_optimal,
                .image = ctx.image,
                .subresource_range = range,
                .src_access_mask = .{ .transfer_write_bit = true },
                .dst_access_mask = .{ .shader_read_bit = true },
                .src_queue_family_index = ctx.queue_family_index,
                .dst_queue_family_index = ctx.queue_family_index,
            };

            vkd().cmdPipelineBarrier(cmd, .{ .transfer_bit = true }, .{ .fragment_shader_bit = true }, .{}, 0, null, 0, null, 1, @ptrCast(&image_barrier));
        }
    };

    try engine.immediateSubmit(ImageCopy{
        .buffer = staging_buffer.handle,
        .image = image.handle,
        .queue_family_index = engine.device.physical_device.graphics_family_index,
        .extent = extent,
    });

    try engine.image_deletion_queue.append(image);

    return image;
}
