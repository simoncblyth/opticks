#version 400

uniform mat4 ModelViewProjection ;
uniform mat4 ModelView ;
uniform mat4 ISNormModelViewProjection ;
uniform vec4 Param ;

layout(location = 0) in vec4  rpos;
layout(location = 1) in ivec4 rflg;  
layout(location = 2) in uvec4 rflq;  
layout(location = 3) in ivec4 rsel;  

out vec4 colour;
out uvec4 flq ;
out ivec4 sel  ;

void main () 
{
    colour = vec4(1.0,1.0,1.0,1.0) ;

    // pass thru to geom.glsl
    flq = rflq ; 
    sel = rsel ; 
    gl_Position = rpos ; 
    gl_PointSize = 1.0;

}


