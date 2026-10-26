// Per-camera post process on the atlas: exposure gain, ego-motion blur, sensor noise.
shader_type canvas_item;
uniform sampler2D src : filter_linear, repeat_disable;
uniform float gain = 1.0;
uniform float noise = 0.0;
uniform vec2 blur = vec2(0.0);   // blur length in UV units
uniform float seed = 0.0;
uniform bool raw = false;

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233)) + seed * 17.13) * 43758.5453); }

void fragment() {
	vec3 c;
	if (raw) {
		c = texture(src, UV).rgb;
	} else {
		c = vec3(0.0);
		for (int i = 0; i < 7; i++) {
			c += texture(src, UV + blur * (float(i) / 6.0 - 0.5)).rgb;
		}
		c = c / 7.0 * gain;
		c += vec3((hash(UV * 997.0) + hash(UV * 1733.0 + 3.1) - 1.0) * noise);
	}
	COLOR = vec4(clamp(c, 0.0, 1.0), 1.0);
}
