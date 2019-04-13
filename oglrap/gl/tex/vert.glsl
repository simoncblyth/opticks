#version 410 core
//#pragma debug(on)

uniform mat4 ModelViewProjection ;
uniform mat4 ModelView ;

layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_colour;
layout(location = 2) in vec3 vertex_normal;
layout(location = 3) in vec2 vertex_texcoord;

out vec3 colour;
out vec2 texcoord;

void main () 
{
    vec4 normal = ModelView * vec4 (vertex_normal, 0.0);

    colour = normalize(vec3(normal))*0.5 + 0.5 ;
    //colour = normalize(vec3(1.))*0.5 + 0.5 ;
    //colour = vertex_colour ;
    //colour = vec3(0,1.0,0) ;

    //gl_Position = ModelViewProjection * vec4 (vertex_position, 1.0);
    gl_Position = vec4 (vertex_position, 1.0);

    texcoord = vertex_texcoord;

}


