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

layout (set = 1, binding = 0) uniform sampler2D tex1;

void main()
{
	const float gamma = 2.2;
	const vec3 pixel = texture(tex1, uv).rgb;

	out_color = vec4(pow(pixel, vec3(1.0 / gamma)), 1.0);
}
