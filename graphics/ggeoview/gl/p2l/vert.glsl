#version 400

// p2l passthrough to geometry shader

uniform mat4 ModelViewProjection ;
uniform mat4 ModelView ;

layout(location = 0) in vec4 vpos ;
layout(location = 1) in vec4 vdir ;

out vec3 colour;
out vec3 direction ; 


void main () 
{
    colour = vec3(1.0,0.0,0.0) ;
    direction = vdir.xyz ;
    gl_Position = vec4( vpos.xyz, 1.0);
}


