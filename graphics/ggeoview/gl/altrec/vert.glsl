#version 400

layout(location = 0) in vec4  rpos;
layout(location = 1) in vec4  rpol;  
layout(location = 2) in ivec4 rflg;  

out vec4 polarization ;

void main () 
{
    // pass thru to geom.glsl
    polarization = rpol ; 
    gl_Position = rpos ; 
    gl_PointSize = 1.0;

}


