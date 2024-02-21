#version 450

layout (location = 0) in vec3 color;
layout (location = 1) in vec2 uv;
layout (location = 2) in vec3 normal;

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
	const vec3 sunlight_dir = normalize(vec3(-1, -1, 1));
	const float light_factor = dot(-sunlight_dir, normalize(normal));
	const vec3 pixel = texture(tex1, uv).rgb * max(light_factor, 0.35);

	const float gamma = 2.2;
	out_color = vec4(pow(pixel, vec3(1.0 / gamma)), 1.0);
}
