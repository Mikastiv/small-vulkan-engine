const std = @import("std");

pub const Vec2 = [2]f32;
pub const Vec3 = [3]f32;
pub const Vec4 = [4]f32;
pub const Mat2 = [2]Vec2;
pub const Mat3 = [3]Vec3;
pub const Mat4 = [4]Vec4;

fn unsupportedType(comptime T: type) void {
    @compileError("unsupported type: " ++ @typeName(T));
}

fn vecsize(comptime T: type) comptime_int {
    return switch (T) {
        Vec2 => 2,
        Vec3 => 3,
        Vec4 => 4,
        else => unsupportedType(T),
    };
}

pub const vec = struct {
    pub inline fn vec2(v: anytype) Vec2 {
        return switch (@TypeOf(v)) {
            Vec3, Vec4 => .{ v[0], v[1] },
            else => unsupportedType(@TypeOf(v)),
        };
    }

    pub inline fn vec3(v: anytype) Vec3 {
        return switch (@TypeOf(v)) {
            Vec2 => .{ v[0], v[1], 0 },
            Vec4 => .{ v[0], v[1], v[2] },
            else => unsupportedType(@TypeOf(v)),
        };
    }

    pub inline fn vec4(v: anytype) Vec4 {
        return switch (@TypeOf(v)) {
            Vec2 => .{ v[0], v[1], 0, 1 },
            Vec3 => .{ v[0], v[1], v[2], 1 },
            else => unsupportedType(@TypeOf(v)),
        };
    }

    pub inline fn vec4Dir(v: anytype) Vec4 {
        return switch (@TypeOf(v)) {
            Vec2 => .{ v[0], v[1], 0, 0 },
            Vec3 => .{ v[0], v[1], v[2], 0 },
            else => unsupportedType(@TypeOf(v)),
        };
    }

    pub inline fn zero(comptime T: type) T {
        const size = vecsize(T);

        var out: T = undefined;
        inline for (0..size) |i| {
            out[i] = 0;
        }

        return out;
    }

    pub inline fn sub(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
        const size = vecsize(@TypeOf(a));

        var out: @TypeOf(a) = undefined;
        inline for (0..size) |i| {
            out[i] = a[i] - b[i];
        }

        return out;
    }

    pub inline fn neg(v: anytype) @TypeOf(v) {
        const size = vecsize(@TypeOf(v));

        var out: @TypeOf(v) = undefined;
        inline for (0..size) |i| {
            out[i] = -v[i];
        }

        return out;
    }

    pub inline fn add(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
        const size = vecsize(@TypeOf(a));

        var out: @TypeOf(a) = undefined;
        inline for (0..size) |i| {
            out[i] = a[i] + b[i];
        }

        return out;
    }

    pub inline fn mul(a: anytype, b: f32) @TypeOf(a) {
        const size = vecsize(@TypeOf(a));

        var out: @TypeOf(a) = undefined;
        inline for (0..size) |i| {
            out[i] = a[i] * b;
        }

        return out;
    }

    pub inline fn div(a: anytype, b: f32) @TypeOf(a) {
        const size = vecsize(@TypeOf(a));

        var out: @TypeOf(a) = undefined;
        inline for (0..size) |i| {
            out[i] = a[i] / b;
        }

        return out;
    }

    pub inline fn length(v: anytype) f32 {
        return @sqrt(length2(v));
    }

    pub inline fn length2(v: anytype) f32 {
        const size = vecsize(@TypeOf(v));

        var out: f32 = 0.0;
        inline for (0..size) |i| {
            out += v[i] * v[i];
        }

        return out;
    }

    pub inline fn unit(v: anytype) @TypeOf(v) {
        const size = vecsize(@TypeOf(v));
        const mag = length(v);

        var out: @TypeOf(v) = undefined;
        inline for (0..size) |i| {
            out[i] = v[i] / mag;
        }

        return out;
    }

    pub inline fn normalize(v: anytype) @TypeOf(v) {
        return unit(v);
    }

    pub inline fn dot(a: anytype, b: @TypeOf(a)) f32 {
        const size = vecsize(@TypeOf(a));

        var out: f32 = 0.0;
        inline for (0..size) |i| {
            out += a[i] * b[i];
        }

        return out;
    }

    pub inline fn cross(a: Vec3, b: Vec3) Vec3 {
        return .{
            (a[1] * b[2]) - (a[2] * b[1]),
            (a[2] * b[0]) - (a[0] * b[2]),
            (a[0] * b[1]) - (a[1] * b[0]),
        };
    }

    pub inline fn distance(a: anytype, b: @TypeOf(a)) f32 {
        return length(sub(a, b));
    }
};

fn matsize(comptime T: type) comptime_int {
    return switch (T) {
        Mat2 => 2,
        Mat3 => 3,
        Mat4 => 4,
        else => unsupportedType(T),
    };
}

fn pointerType(comptime T: type) type {
    if (@typeInfo(T) != .Pointer) {
        @compileError("only accepts pointers");
    }
    return @typeInfo(T).Pointer.child;
}

fn checkSamePointerType(comptime A: type, comptime B: type) void {
    if (pointerType(A) != pointerType(B)) {
        @compileError("a and b must be the same type");
    }
}

pub const mat = struct {
    pub inline fn mat2(m: anytype) Mat2 {
        const T = pointerType(@TypeOf(m));
        return switch (T) {
            Mat3, Mat4 => .{
                .{ m[0][0], m[0][1] },
                .{ m[1][0], m[1][1] },
            },
            else => unsupportedType(T),
        };
    }

    pub inline fn mat3(m: anytype) Mat3 {
        const T = pointerType(@TypeOf(m));
        return switch (T) {
            Mat2 => .{
                .{ m[0][0], m[0][1], 0 },
                .{ m[1][0], m[1][1], 0 },
                .{ 0, 0, 0 },
            },
            Mat4 => .{
                .{ m[0][0], m[0][1], m[0][2] },
                .{ m[1][0], m[1][1], m[1][2] },
                .{ m[2][0], m[2][1], m[2][2] },
            },
            else => unsupportedType(T),
        };
    }

    pub inline fn mat4(m: anytype) Mat4 {
        const T = pointerType(@TypeOf(m));
        return switch (T) {
            Mat2 => .{
                .{ m[0][0], m[0][1], 0, 0 },
                .{ m[1][0], m[1][1], 0, 0 },
                .{ 0, 0, 0, 0 },
                .{ 0, 0, 0, 0 },
            },
            Mat3 => .{
                .{ m[0][0], m[0][1], m[0][2], 0 },
                .{ m[1][0], m[1][1], m[1][2], 0 },
                .{ m[2][0], m[2][1], m[2][2], 0 },
                .{ 0, 0, 0, 0 },
            },
            else => unsupportedType(T),
        };
    }

    pub inline fn identity(comptime T: type) T {
        var out = std.mem.zeroes(T);

        switch (T) {
            Mat2 => {
                out[0][0] = 1;
                out[1][1] = 1;
            },
            Mat3 => {
                out[0][0] = 1;
                out[1][1] = 1;
                out[2][2] = 1;
            },
            Mat4 => {
                out[0][0] = 1;
                out[1][1] = 1;
                out[2][2] = 1;
                out[3][3] = 1;
            },
            else => unsupportedType(T),
        }

        return out;
    }

    pub inline fn zero(comptime T: type) T {
        const size = matsize(T);

        var out: T = undefined;
        inline for (0..size) |j| {
            inline for (0..size) |i| {
                out[j][i] = 0;
            }
        }

        return out;
    }

    pub fn add(a: anytype, b: anytype) pointerType(@TypeOf(a)) {
        checkSamePointerType(@TypeOf(a), @TypeOf(b));

        const T = pointerType(@TypeOf(a));
        const size = matsize(T);

        var out: T = undefined;
        inline for (0..size) |i| {
            out[i] = vec.add(a[i], b[i]);
        }

        return out;
    }

    pub fn sub(a: anytype, b: anytype) pointerType(@TypeOf(a)) {
        checkSamePointerType(@TypeOf(a), @TypeOf(b));

        const T = pointerType(@TypeOf(a));
        const size = matsize(T);

        var out: T = undefined;
        inline for (0..size) |i| {
            out[i] = vec.sub(a[i], b[i]);
        }

        return out;
    }

    pub fn divScalar(a: anytype, b: f32) pointerType(@TypeOf(a)) {
        const T = pointerType(@TypeOf(a));
        const size = matsize(T);

        var out: T = undefined;
        for (0..size) |i| {
            out[i] = vec.div(a[i], b);
        }

        return out;
    }

    pub fn mulScalar(a: anytype, b: f32) pointerType(@TypeOf(a)) {
        const T = pointerType(@TypeOf(a));
        const size = matsize(T);

        var out: T = undefined;
        inline for (0..size) |i| {
            out[i] = vec.mul(a[i], b);
        }

        return out;
    }

    pub fn mulVec(m: anytype, v: anytype) @TypeOf(v) {
        const T = pointerType(@TypeOf(m));
        const mat_size = matsize(T);
        const vec_size = vecsize(@TypeOf(v));
        if (mat_size != vec_size) @compileError("incompatible matrix and vector");

        var out: @TypeOf(v) = undefined;
        inline for (0..mat_size) |row| {
            out[row] = 0;
            inline for (0..mat_size) |col| {
                out[row] += m[col][row] * v[col];
            }
        }

        return out;
    }

    pub fn mul(a: anytype, b: anytype) pointerType(@TypeOf(a)) {
        checkSamePointerType(@TypeOf(a), @TypeOf(b));

        const T = pointerType(@TypeOf(a));
        const size = matsize(T);

        var out: T = undefined;
        inline for (0..size) |row| {
            inline for (0..size) |col| {
                const v = b[row];
                out[row][col] = 0;
                inline for (0..size) |i| {
                    out[row][col] += a[i][col] * v[i];
                }
            }
        }

        return out;
    }

    pub fn transpose(m: anytype) pointerType(@TypeOf(m)) {
        const T = pointerType(@TypeOf(m));
        const size = matsize(T);

        var out: T = undefined;
        inline for (0..size) |row| {
            inline for (0..size) |col| {
                out[row][col] = m[col][row];
            }
        }

        return out;
    }

    pub inline fn scaling2d(s: Vec2) Mat3 {
        var out = std.mem.zeroes(Mat3);
        out[0][0] = s[0];
        out[1][1] = s[1];
        out[2][2] = 1;
        return out;
    }

    pub inline fn scaling2dScalar(s: f32) Mat3 {
        return scaling2d(.{ s, s });
    }

    pub inline fn scale2d(m: *const Mat3, s: Vec2) Mat3 {
        var out: Mat3 = undefined;
        out[0] = vec.mul(m[0], s[0]);
        out[1] = vec.mul(m[1], s[1]);
        out[2] = m[2];
        return out;
    }

    pub inline fn scale2dScalar(m: *const Mat3, s: f32) Mat3 {
        return scale2d(m, .{ s, s });
    }

    pub inline fn scaling(s: Vec3) Mat4 {
        var out = std.mem.zeroes(Mat4);
        out[0][0] = s[0];
        out[1][1] = s[1];
        out[2][2] = s[2];
        out[3][3] = 1;
        return out;
    }

    pub inline fn scalingScalar(s: f32) Mat4 {
        return scaling(.{ s, s, s });
    }

    pub fn scale(m: *const Mat4, s: Vec3) Mat4 {
        var out: Mat4 = undefined;
        out[0] = vec.mul(m[0], s[0]);
        out[1] = vec.mul(m[1], s[1]);
        out[2] = vec.mul(m[2], s[2]);
        out[3] = m[3];
        return out;
    }

    pub fn scaleScalar(m: *const Mat4, s: f32) Mat4 {
        return scale(m, .{ s, s, s });
    }

    pub inline fn translation2d(t: Vec2) Mat3 {
        var out = identity(Mat3);
        out[2][0] = t[0];
        out[2][1] = t[1];
        return out;
    }

    pub fn translate2d(m: *const Mat3, t: Vec2) Mat3 {
        var out = m.*;
        const a = vec.mul(m[0], t[0]);
        const b = vec.mul(m[1], t[1]);
        out[2] = vec.add(vec.add(a, b), m[2]);
        return out;
    }

    pub inline fn translation(t: Vec3) Mat4 {
        var out = identity(Mat4);
        out[3][0] = t[0];
        out[3][1] = t[1];
        out[3][2] = t[2];
        return out;
    }

    pub fn translate(m: *const Mat4, t: Vec3) Mat4 {
        var out = m.*;
        const a = vec.mul(m[0], t[0]);
        const b = vec.mul(m[1], t[1]);
        const c = vec.mul(m[2], t[2]);
        out[3] = vec.add(vec.add(a, b), vec.add(c, m[3]));
        return out;
    }

    pub inline fn rotation2d(angle: f32) Mat3 {
        const s = @sin(angle);
        const c = @cos(angle);

        var out = identity(Mat3);
        out[0][0] = c;
        out[0][1] = s;
        out[1][0] = -s;
        out[1][1] = c;
        return out;
    }

    pub fn rotate2d(m: *const Mat2, angle: f32) Mat3 {
        const rot = rotation2d(angle);

        var out = zero(Mat3);

        var a = vec.mul(m[0], rot[0][0]);
        var b = vec.mul(m[1], rot[0][1]);
        out[0] = vec.add(a, b);

        a = vec.mul(m[0], rot[1][0]);
        b = vec.mul(m[1], rot[1][1]);
        out[1] = vec.add(a, b);

        out[2] = m[2];

        return out;
    }

    pub inline fn rotation(angle: f32, axis: Vec3) Mat4 {
        const s = @sin(angle);
        const c = @cos(angle);
        const a = vec.unit(axis);
        const t = vec.mul(a, 1 - c);

        var out = identity(Mat4);
        out[0][0] = c + t[0] * a[0];
        out[0][1] = t[0] * a[1] + s * a[2];
        out[0][2] = t[0] * a[2] - s * a[1];
        out[1][0] = t[1] * a[0] - s * a[2];
        out[1][1] = c + t[1] * a[1];
        out[1][2] = t[1] * a[2] + s * a[0];
        out[2][0] = t[2] * a[0] + s * a[1];
        out[2][1] = t[2] * a[1] - s * a[0];
        out[2][2] = c + t[2] * a[2];
        return out;
    }

    pub fn rotate(m: *const Mat4, angle: f32, axis: Vec3) Mat4 {
        const rot = rotation(angle, axis);

        var out: Mat4 = undefined;

        var a = vec.mul(m[0], rot[0][0]);
        var b = vec.mul(m[1], rot[0][1]);
        var c = vec.mul(m[2], rot[0][2]);
        out[0] = vec.add(vec.add(a, b), c);

        a = vec.mul(m[0], rot[1][0]);
        b = vec.mul(m[1], rot[1][1]);
        c = vec.mul(m[2], rot[1][2]);
        out[1] = vec.add(vec.add(a, b), c);

        a = vec.mul(m[0], rot[2][0]);
        b = vec.mul(m[1], rot[2][1]);
        c = vec.mul(m[2], rot[2][2]);
        out[2] = vec.add(vec.add(a, b), c);

        out[3] = m[3];

        return out;
    }

    pub inline fn orthographic(left: f32, right: f32, top: f32, bottom: f32, near: f32, far: f32) Mat4 {
        var out = std.mem.zeroes(Mat4);
        out[0][0] = 2 / (right - left);
        out[1][1] = 2 / (bottom - top);
        out[2][2] = 1 / (far - near);
        out[3][0] = -(right + left) / (right - left);
        out[3][1] = -(bottom + top) / (bottom - top);
        out[3][2] = -near / (far - near);
        return out;
    }

    pub inline fn perspective(fovy: f32, aspect: f32, near: f32, far: f32) Mat4 {
        std.debug.assert(near > 0 and far > 0);

        const g = 1.0 / @tan(fovy / 2.0);
        const k = far / (far - near);

        var out = std.mem.zeroes(Mat4);
        out[0][0] = g / aspect;
        out[1][1] = -g;
        out[2][2] = -k;
        out[2][3] = -1;
        out[3][2] = -near * k;
        return out;
    }

    pub fn lookAtDir(eye: Vec3, dir: Vec3, up: Vec3) Mat4 {
        std.debug.assert(vec.length2(dir) != 0);

        const w = vec.normalize(dir);
        const u = vec.normalize(vec.cross(up, w));
        const v = vec.cross(w, u);

        // const dot_u = vec.dot(u, eye);
        // const dot_v = vec.dot(v, eye);
        // const dot_w = vec.dot(w, eye);

        const t = translation(vec.neg(eye));
        const r: Mat4 = .{
            .{ u[0], v[0], w[0], 0 },
            .{ u[1], v[1], w[1], 0 },
            .{ u[2], v[2], w[2], 0 },
            .{ 0, 0, 0, 1 },
        };

        // return .{
        //     .{ u[0], v[0], w[0], 0 },
        //     .{ u[1], v[1], w[1], 0 },
        //     .{ u[2], v[2], w[2], 0 },
        //     .{ -dot_u, -dot_v, -dot_w, 1 },
        // };
        return mul(&t, &r);
    }

    pub fn lookAt(eye: Vec3, target: Vec3, up: Vec3) Mat4 {
        return lookAtDir(eye, vec.sub(eye, target), up);
    }

    pub fn determinant(m: anytype) f32 {
        const T = pointerType(@TypeOf(m));

        return switch (T) {
            Mat2 => m[0][0] * m[1][1] - m[1][0] * m[0][1],
            Mat3 => blk: {
                const cofactor0 = m[1][1] * m[2][2] - m[2][1] * m[1][2];
                const cofactor1 = m[0][1] * m[2][2] - m[2][1] * m[0][2];
                const cofactor2 = m[0][1] * m[1][2] - m[1][1] * m[0][2];

                break :blk m[0][0] * cofactor0 - m[1][0] * cofactor1 + m[2][0] * cofactor2;
            },
            Mat4 => blk: {
                const subfactor0 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
                const subfactor1 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
                const subfactor2 = m[1][2] * m[2][3] - m[2][2] * m[1][3];
                const subfactor3 = m[0][2] * m[3][3] - m[3][2] * m[0][3];
                const subfactor4 = m[0][2] * m[2][3] - m[2][2] * m[0][3];
                const subfactor5 = m[0][2] * m[1][3] - m[1][2] * m[0][3];

                const cofactor0 = m[1][1] * subfactor0 - m[2][1] * subfactor1 + m[3][1] * subfactor2;
                const cofactor1 = m[0][1] * subfactor0 - m[2][1] * subfactor3 + m[3][1] * subfactor4;
                const cofactor2 = m[0][1] * subfactor1 - m[1][1] * subfactor3 + m[3][1] * subfactor5;
                const cofactor3 = m[0][1] * subfactor2 - m[1][1] * subfactor4 + m[2][1] * subfactor5;

                break :blk m[0][0] * cofactor0 -
                    m[1][0] * cofactor1 +
                    m[2][0] * cofactor2 -
                    m[3][0] * cofactor3;
            },
            else => unsupportedType(T),
        };
    }

    pub fn inverseTranspose(m: anytype) pointerType(@TypeOf(m)) {
        const T = pointerType(@TypeOf(m));

        return switch (T) {
            Mat2 => blk: {
                const det = determinant(m);

                const inverse = Mat2{
                    .{ m[1][1], -m[0][1] },
                    .{ -m[1][0], m[0][0] },
                };

                break :blk divScalar(&inverse, det);
            },
            Mat3 => blk: {
                const det = determinant(m);

                const inverse = Mat3{
                    .{
                        m[1][1] * m[2][2] - m[2][1] * m[1][2],
                        -(m[1][0] * m[2][2] - m[2][0] * m[1][2]),
                        m[1][0] * m[2][1] - m[2][0] * m[1][1],
                    },
                    .{
                        -(m[0][1] * m[2][2] - m[2][1] * m[0][2]),
                        m[0][0] * m[2][2] - m[2][0] * m[0][2],
                        -(m[0][0] * m[2][1] - m[2][0] * m[0][1]),
                    },
                    .{
                        m[0][1] * m[1][2] - m[1][1] * m[0][2],
                        -(m[0][0] * m[1][2] - m[1][0] * m[0][2]),
                        m[0][0] * m[1][1] - m[1][0] * m[0][1],
                    },
                };

                break :blk divScalar(&inverse, det);
            },
            Mat4 => blk: {
                const subfactor00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
                const subfactor01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
                const subfactor02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
                const subfactor03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
                const subfactor04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
                const subfactor05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
                const subfactor06 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
                const subfactor07 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
                const subfactor08 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
                const subfactor09 = m[1][0] * m[3][3] - m[3][0] * m[1][3];
                const subfactor10 = m[1][0] * m[3][2] - m[3][0] * m[1][2];
                const subfactor11 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
                const subfactor12 = m[1][2] * m[2][3] - m[2][2] * m[1][3];
                const subfactor13 = m[1][1] * m[2][3] - m[2][1] * m[1][3];
                const subfactor14 = m[1][1] * m[2][2] - m[2][1] * m[1][2];
                const subfactor15 = m[1][0] * m[2][3] - m[2][0] * m[1][3];
                const subfactor16 = m[1][0] * m[2][2] - m[2][0] * m[1][1];
                const subfactor17 = m[1][0] * m[2][1] - m[2][0] * m[1][1];

                const inverse = Mat4{
                    .{
                        m[1][1] * subfactor00 - m[1][2] * subfactor01 + m[1][3] * subfactor02,
                        -(m[1][0] * subfactor00 - m[1][2] * subfactor03 + m[1][3] * subfactor04),
                        m[1][0] * subfactor01 - m[1][1] * subfactor03 + m[1][3] * subfactor05,
                        -(m[1][0] * subfactor02 - m[1][1] * subfactor04 + m[1][2] * subfactor05),
                    },
                    .{
                        -(m[0][1] * subfactor00 - m[0][2] * subfactor01 + m[0][3] * subfactor02),
                        m[0][0] * subfactor00 - m[0][2] * subfactor03 + m[0][3] * subfactor04,
                        -(m[0][0] * subfactor01 - m[0][1] * subfactor03 + m[0][3] * subfactor05),
                        m[0][0] * subfactor02 - m[0][1] * subfactor04 + m[0][2] * subfactor05,
                    },
                    .{
                        m[0][1] * subfactor06 - m[0][2] * subfactor07 + m[0][3] * subfactor08,
                        -(m[0][0] * subfactor06 - m[0][2] * subfactor09 + m[0][3] * subfactor10),
                        m[0][0] * subfactor07 - m[0][1] * subfactor09 + m[0][3] * subfactor11,
                        -(m[0][0] * subfactor08 - m[0][1] * subfactor10 + m[0][2] * subfactor11),
                    },
                    .{
                        -(m[0][1] * subfactor12 - m[0][2] * subfactor13 + m[0][3] * subfactor14),
                        m[0][0] * subfactor12 - m[0][2] * subfactor15 + m[0][3] * subfactor16,
                        -(m[0][0] * subfactor13 - m[0][1] * subfactor15 + m[0][3] * subfactor17),
                        m[0][0] * subfactor14 - m[0][1] * subfactor16 + m[0][2] * subfactor17,
                    },
                };

                const det =
                    m[0][0] * inverse[0][0] +
                    m[0][1] * inverse[0][1] +
                    m[0][2] * inverse[0][2] +
                    m[0][3] * inverse[0][3];

                break :blk divScalar(&inverse, det);
            },
            else => unsupportedType(T),
        };
    }
};

pub const Sphere = struct {
    center: Vec3,
    radius: f32,

    pub fn contains(self: @This(), point: Vec3) bool {
        const tolerance = 0.0001;
        return vec.distance(self.center, point) <= self.radius + tolerance;
    }

    pub fn circumsphere(a: Vec3, b: Vec3, c: Vec3, d: Vec3) @This() {
        const a2 = vec.length2(a);
        const b2 = vec.length2(b);
        const c2 = vec.length2(c);
        const d2 = vec.length2(d);

        const detInv = 1.0 / mat.determinant(Mat4{
            .{ a[0], a[1], a[2], 1 },
            .{ b[0], b[1], b[2], 1 },
            .{ c[0], c[1], c[2], 1 },
            .{ d[0], d[1], d[2], 1 },
        });

        const x = mat.determinant(Mat4{
            .{ a2, a[1], a[2], 1 },
            .{ b2, b[1], b[2], 1 },
            .{ c2, c[1], c[2], 1 },
            .{ d2, d[1], d[2], 1 },
        });
        const y = mat.determinant(Mat4{
            .{ a[0], a2, a[2], 1 },
            .{ b[0], b2, b[2], 1 },
            .{ c[0], c2, c[2], 1 },
            .{ d[0], d2, d[2], 1 },
        });
        const z = mat.determinant(Mat4{
            .{ a[0], a[1], a2, 1 },
            .{ b[0], b[1], b2, 1 },
            .{ c[0], c[1], c2, 1 },
            .{ d[0], d[1], d2, 1 },
        });

        const center = vec.mul(Vec3{ x, y, z }, detInv * 0.5);

        return .{
            .center = center,
            .radius = vec.distance(center, a),
        };
    }

    pub fn circumsphereTriangle(a: Vec3, b: Vec3, c: Vec3) @This() {
        const ca = vec.sub(c, a);
        const ba = vec.sub(b, a);
        const crs = vec.cross(ba, ca);

        const t0 = vec.mul(vec.cross(crs, ba), vec.length2(ca));
        const t1 = vec.mul(vec.cross(ca, crs), vec.length2(ba));
        const x = vec.add(t0, t1);

        const rvec = vec.div(x, 2.0 * vec.length2(crs));

        return .{
            .center = vec.add(a, rvec),
            .radius = vec.length(rvec),
        };
    }

    pub fn fromDiameter(a: Vec3, b: Vec3) @This() {
        const center = vec.mul(vec.add(a, b), 0.5);
        return .{
            .center = center,
            .radius = vec.distance(center, a),
        };
    }
};

fn smallestEnclosingSphereImpl(points: []Vec3, end: usize, pin1: ?Vec3, pin2: ?Vec3, pin3: ?Vec3) Sphere {
    var sphere: Sphere = undefined;

    var current: usize = 0;
    if (pin1 != null and pin2 != null and pin3 != null) {
        sphere = Sphere.circumsphereTriangle(pin1.?, pin2.?, pin3.?);
    } else if (pin1 != null and pin2 != null) {
        sphere = Sphere.fromDiameter(pin1.?, pin2.?);
    } else if (pin1 != null) {
        sphere = Sphere.fromDiameter(points[current], pin1.?);
        current += 1;
    } else {
        sphere = Sphere.fromDiameter(points[current], points[current + 1]);
        current += 2;
    }

    while (current < end) {
        if (!sphere.contains(points[current])) {
            if (pin1 != null and pin2 != null and pin3 != null) {
                sphere = Sphere.circumsphere(pin1.?, pin2.?, pin3.?, points[current]);
            } else if (pin1 != null and pin2 != null) {
                sphere = smallestEnclosingSphereImpl(points, current, pin1, pin2, points[current]);
            } else if (pin1 != null) {
                sphere = smallestEnclosingSphereImpl(points, current, pin1, points[current], null);
            } else {
                sphere = smallestEnclosingSphereImpl(points, current, points[current], null, null);
            }
        }
        current += 1;
    }

    return sphere;
}

pub fn smallestEnclosingSphere(points: []Vec3) Sphere {
    std.debug.assert(points.len > 1);
    return smallestEnclosingSphereImpl(points, points.len, null, null, null);
}
