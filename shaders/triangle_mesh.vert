#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec3 color;
layout (location = 3) in vec2 uv;

layout (location = 0) out vec3 out_color;
layout (location = 1) out vec2 out_uv;
layout (location = 2) out vec3 out_normal;

layout (set = 0, binding = 0) uniform GlobalData {
	mat4 view;
    mat4 proj;
    mat4 view_proj;
	vec4 fog_color;
	vec4 fog_distances;
	vec4 ambient_color;
	vec4 sunlight_direction;
	vec4 sunlight_color;
} global_data;

layout (push_constant) uniform Constants {
    mat4 model_matrix;
} push;

void main() {
    mat4 transform_matrix = global_data.view_proj * push.model_matrix;
    gl_Position = transform_matrix * vec4(position, 1);
    out_color = color;
    out_uv = uv;
    out_normal = normal;
}
