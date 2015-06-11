#version 400

uniform mat4 ModelViewProjection ;
uniform mat4 ModelView ;
uniform vec4 ClipPlane ;
uniform vec4 Param ;

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_colour;
layout(location = 2) in vec3 vertex_normal;
layout(location = 3) in vec2 vertex_texcoord;

float gl_ClipDistance[1]; 

out vec4 colour;
out vec2 texcoord;

void main () 
{
    vec4 normal = ModelView * vec4 (vertex_normal, 0.0);

    gl_ClipDistance[0] = dot(vec4(vertex_position, 1.0), ClipPlane);

    colour = vec4( normalize(vec3(normal))*0.5 + 0.5, Param.z ) ;

    gl_Position = ModelViewProjection * vec4 (vertex_position, 1.0);


    texcoord = vertex_texcoord;

}


