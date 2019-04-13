#version 410 core

// altrec/vert.glsl

layout(location = 0) in vec4  rpos;
layout(location = 1) in vec4  rpol;  
layout(location = 2) in ivec4 rflg;  
layout(location = 3) in ivec4 rsel;  
layout(location = 4) in uvec4 rflq;  

out vec4 polarization ;
out uvec4 flags ;
out uvec4 flq ;
out ivec4 sel  ;

void main () 
{
    sel = rsel ; 
    polarization = rpol ; 
    flags = rflg ; 
    flq = rflq ;

    gl_Position = rpos ; 
    gl_PointSize = 1.0;

}


