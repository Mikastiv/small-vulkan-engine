const vk = @import("vulkan-zig");

pub const device = vk.DeviceCommandFlags{
    .destroyDevice = true,
    .getDeviceQueue = true,
    .createSwapchainKHR = true,
    .destroySwapchainKHR = true,
    .getSwapchainImagesKHR = true,
    .createImageView = true,
    .destroyImageView = true,
    .createRenderPass = true,
    .destroyRenderPass = true,
    .createFramebuffer = true,
    .destroyFramebuffer = true,
    .createSemaphore = true,
    .destroySemaphore = true,
    .createFence = true,
    .destroyFence = true,
    .createShaderModule = true,
    .destroyShaderModule = true,
    .createPipelineLayout = true,
    .destroyPipelineLayout = true,
    .createGraphicsPipelines = true,
    .destroyPipeline = true,
    .createDescriptorSetLayout = true,
    .destroyDescriptorSetLayout = true,
    .createDescriptorPool = true,
    .destroyDescriptorPool = true,
    .allocateDescriptorSets = true,
    .updateDescriptorSets = true,
    .createCommandPool = true,
    .destroyCommandPool = true,
    .resetCommandPool = true,
    .allocateCommandBuffers = true,
    .freeCommandBuffers = true,
    .resetCommandBuffer = true,
    .beginCommandBuffer = true,
    .waitForFences = true,
    .deviceWaitIdle = true,
    .resetFences = true,
    .acquireNextImageKHR = true,
    .queueSubmit = true,
    .queuePresentKHR = true,
    .endCommandBuffer = true,
    .createSampler = true,
    .cmdBeginRenderPass = true,
    .cmdSetViewport = true,
    .cmdSetScissor = true,
    .cmdBindPipeline = true,
    .cmdDraw = true,
    .cmdEndRenderPass = true,
    .cmdBindVertexBuffers = true,
    .cmdPushConstants = true,
    .cmdBindDescriptorSets = true,
    .cmdCopyBuffer = true,
    .cmdPipelineBarrier = true,
    .cmdCopyBufferToImage = true,
};
