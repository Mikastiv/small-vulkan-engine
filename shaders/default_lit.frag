#version 450

layout (location = 0) in vec3 in_color;

layout (location = 0) out vec4 out_color;

layout (set = 0, binding = 1) uniform SceneData {
	vec4 fog_color;
	vec4 fog_distances;
	vec4 ambient_color;
	vec4 sunlight_direction;
	vec4 sunlight_color;
} scene_data;

void main()
{
	out_color = vec4(in_color + scene_data.ambient_color.rgb, 1.0);
}
