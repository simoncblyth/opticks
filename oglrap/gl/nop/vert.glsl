#version 400

//  nop/vert.glsl

layout(location = 0) in vec4 vpos;
layout(location = 1) in vec4 vdir;  
layout(location = 2) in vec4 vpol;

out vec4 polarization ;

void main () 
{
    polarization = vpol ; 
    gl_Position = vpos  ; 
}

