#version 400

// pass thru to geom.glsl

layout(location = 0) in vec4  rpos;
layout(location = 1) in vec4  rpol;  
layout(location = 2) in uvec4 rflg;   

out vec4 polarization ;
out uvec4 flags ;

void main () 
{
    polarization = rpol ; 
    flags = rflg ; 
    gl_Position = rpos ; 
    gl_PointSize = 1.0;
}


