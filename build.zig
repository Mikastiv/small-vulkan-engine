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
        .enable_validation = if (optimize == .Debug) true else false,
    });

    const glfw = b.dependency("glfw", .{
        .target = target,
        .optimize = .ReleaseFast,
    }).artifact("glfw");

    const vulkan_lib = if (target.result.os.tag == .windows) "vulkan-1" else "vulkan";
    const vulkan_sdk = b.graph.env_map.get("VK_SDK_PATH") orelse @panic("VK_SDK_PATH is not set");
    const shaders = vkgen.ShaderCompileStep.create(b, &.{ "glslc", "--target-env=vulkan1.1" }, "-o");
    shaders.add("triangle_vert", shader_base_path ++ "triangle.vert", .{});
    shaders.add("triangle_frag", shader_base_path ++ "triangle.frag", .{});
    shaders.add("triangle_mesh_vert", shader_base_path ++ "triangle_mesh.vert", .{});
    shaders.add("default_lit", shader_base_path ++ "default_lit.frag", .{});
    shaders.add("textured_lit", shader_base_path ++ "textured_lit.frag", .{});
    shaders.add("widget_3d_vert", shader_base_path ++ "widget_3d.vert", .{});
    shaders.add("widget_3d_frag", shader_base_path ++ "widget_3d.frag", .{});

    const wf = b.addWriteFiles();
    const stb_image = wf.add("stb_image.c",
        \\#define STB_IMAGE_IMPLEMENTATION
        \\#include "stb_image.h"
    );

    const cimgui = b.addStaticLibrary(.{
        .name = "cimgui",
        .target = target,
        .optimize = .ReleaseFast,
    });
    cimgui.linkLibCpp();
    cimgui.linkLibrary(glfw);
    cimgui.addIncludePath(.{ .path = "lib/cimgui/imgui" });
    cimgui.addCSourceFiles(.{
        .files = &.{
            "lib/cimgui/cimgui.cpp",
            "lib/cimgui/cimgui_impl_glfw.cpp",
            "lib/cimgui/cimgui_impl_vulkan.cpp",
            "lib/cimgui/imgui/imgui.cpp",
            "lib/cimgui/imgui/imgui_demo.cpp",
            "lib/cimgui/imgui/imgui_draw.cpp",
            "lib/cimgui/imgui/imgui_impl_glfw.cpp",
            "lib/cimgui/imgui/imgui_impl_vulkan.cpp",
            "lib/cimgui/imgui/imgui_tables.cpp",
            "lib/cimgui/imgui/imgui_widgets.cpp",
        },
    });

    const exe = b.addExecutable(.{
        .name = "vulkan_guide",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    exe.root_module.addImport("vk-kickstart", vk_kickstart.module("vk-kickstart"));
    exe.root_module.addImport("vulkan-zig", vkzig.module("vulkan-zig"));
    exe.root_module.addImport("shaders", shaders.getModule());
    exe.linkLibCpp();
    exe.linkLibrary(glfw);
    exe.linkLibrary(cimgui);
    exe.addIncludePath(.{ .path = b.pathJoin(&.{ vulkan_sdk, "include" }) });
    exe.addLibraryPath(.{ .path = b.pathJoin(&.{ vulkan_sdk, "lib" }) });
    exe.linkSystemLibrary(vulkan_lib);
    exe.addIncludePath(.{ .path = "lib/cimgui" });
    exe.addIncludePath(.{ .path = "lib/cimgui/imgui" });
    exe.addIncludePath(.{ .path = "lib/stb_image" });
    exe.addCSourceFile(.{ .file = stb_image });

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
