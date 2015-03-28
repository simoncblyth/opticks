#version 400

uniform mat4 ModelViewProjection ;

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_colour;
out vec3 colour;

void main () 
{
    colour = vertex_colour;
    gl_Position = ModelViewProjection * vec4 (vertex_position, 1.0);
    //gl_Position = vec4 (vertex_position, 1.0);
}


