#version 450

layout (location = 0) in vec3 color;
layout (location = 1) in vec2 uv;

layout (location = 0) out vec4 out_color;

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

layout (set = 2, binding = 0) uniform sampler2D tex1;

void main()
{
	out_color = vec4(texture(tex1, uv).rgb, 1.0);
}
