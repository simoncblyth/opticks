#version 410 core

in vec3 fcolour;
out vec4 frag_colour;

void main ()  
{
   frag_colour = vec4(fcolour, 1.0);
}

