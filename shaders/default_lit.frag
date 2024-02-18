#version 450

layout (location = 0) in vec3 in_color;

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

void main()
{
	out_color = vec4(in_color + global_data.ambient_color.rgb, 1.0);
}
