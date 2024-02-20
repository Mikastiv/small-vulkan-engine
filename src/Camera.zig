const std = @import("std");
const math = @import("math.zig");

pos: math.Vec3,
right: math.Vec3,
dir: math.Vec3,
pitch: f32,
yaw: f32,

pub fn init(pos: math.Vec3) @This() {
    var self: @This() = .{
        .pos = pos,
        .right = .{ 0, 0, 0 },
        .dir = .{ 0, 0, 0 },
        .pitch = 0,
        .yaw = 180,
    };

    self.update(.{ 0, 0 });
    return self;
}

pub fn update(self: *@This(), offset: math.Vec2) void {
    self.yaw += offset[0];
    self.pitch += offset[1];

    if (self.pitch > 89) self.pitch = 89;
    if (self.pitch < -89) self.pitch = -89;

    // const pitch = std.math.degreesToRadians(f32, self.pitch);
    // const yaw = std.math.degreesToRadians(f32, self.yaw);

    // const dir = math.Vec3{
    //     @sin(yaw) * @cos(pitch),
    //     @sin(pitch),
    //     -@cos(yaw) * @cos(pitch),
    // };

    const dir = math.vec.sub(self.pos, .{ 0, 6, 0 });
    const up: math.Vec3 = .{ 0, 1, 0 };
    self.dir = math.vec.normalize(dir);
    self.right = math.vec.normalize(math.vec.cross(self.dir, up));
}

pub fn viewMatrix(self: *const @This()) math.Mat4 {
    return math.mat.lookAtDir(self.pos, self.dir, .{ 0, 1, 0 });
}
