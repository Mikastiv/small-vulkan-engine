const std = @import("std");
const vkgen = @import("vulkan_zig");

const shader_base_path = "shaders/";

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const xml_path: []const u8 = b.pathFromRoot("vk.xml");

    const vkzig = b.dependency("vulkan_zig", .{
        .registry = xml_path,
    });

    const vk_kickstart = b.dependency("vk_kickstart", .{
        .registry = xml_path,
    });

    const glfw = b.dependency("glfw", .{
        .target = target,
        .optimize = .ReleaseFast,
    });

    const shaders = vkgen.ShaderCompileStep.create(b, &.{ "glslc", "--target-env=vulkan1.1" }, "-o");

    const exe = b.addExecutable(.{
        .name = "vulkan_guide",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    exe.root_module.addImport("vk-kickstart", vk_kickstart.module("vk-kickstart"));
    exe.root_module.addImport("vulkan-zig", vkzig.module("vulkan-zig"));
    exe.root_module.addImport("shaders", shaders.getModule());
    exe.linkLibrary(glfw.artifact("glfw"));

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const exe_unit_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });
    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_exe_unit_tests.step);
}
