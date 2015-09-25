#version 400

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_colour;
layout(location = 2) in vec3 vertex_normal;

void main () 
{
    gl_Position = vec4( vertex_position, 1.0)  ; 
}


