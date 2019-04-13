#version 410 core

// dbg/vert.glsl

uniform mat4 ModelViewProjection ;
uniform mat4 ModelView ;

layout(location = 0) in vec4 vpos;
layout(location = 1) in vec4 vdir;
layout(location = 2) in vec4 vpol;

out vec4 colour;

void main () 
{
    colour = vec4(1.0,1.0,1.0,1.0) ;
    gl_Position = ModelViewProjection * vec4( vec3(vpos), 1.0 );
    gl_PointSize = 5.0;

}


