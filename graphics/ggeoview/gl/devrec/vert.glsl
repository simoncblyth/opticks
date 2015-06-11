#version 400

layout(location = 0) in vec4  rpos;
layout(location = 1) in vec4  rpol;  
layout(location = 2) in ivec4 rflg;   
// TODO: better to use uvec4 for bitfields

out vec4 polarization ;
out ivec4 flags ;

void main () 
{
    // pass thru to geom.glsl
    polarization = rpol ; 
    flags = rflg ; 
    gl_Position = rpos ; 
    gl_PointSize = 1.0;

}


