#version 400

in vec3 colour;
//in vec2 texcoord;

out vec4 frag_colour;

void main () 
{
   frag_colour = vec4 (colour, 1.0);

}


