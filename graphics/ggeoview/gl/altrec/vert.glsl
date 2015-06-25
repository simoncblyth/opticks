#version 400

layout(location = 0) in vec4  rpos;
layout(location = 1) in vec4  rpol;  
layout(location = 2) in ivec4 rflg;  
layout(location = 3) in ivec4 rsel;  

out vec4 polarization ;
out ivec4 sel  ;

void main () 
{
    // pass thru to geom.glsl
    sel = rsel ; 
    polarization = rpol ; 
    gl_Position = rpos ; 
    gl_PointSize = 1.0;

}


