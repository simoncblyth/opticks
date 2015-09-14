#version 400 core
#pragma debug(on)

in vec3 colour;
in vec2 texcoord;

out vec4 frag_colour;
uniform sampler2D texSampler;

void main () 
{
   frag_colour = texture(texSampler, texcoord);

   //frag_colour = vec4 (colour, 1.0);
   //frag_colour = vec4 (1.0,0.0,0.0, 1.0);
}


