#version 400

uniform mat4 ModelViewProjection ;
uniform mat4 ModelView ;
uniform mat4 ISNormModelViewProjection ;
uniform vec4 Param ;

layout(location = 0) in vec4  rpos;
layout(location = 1) in ivec4 rflg;  
layout(location = 2) in ivec4 rflq;  

out vec4 colour;
out ivec4 flq ;

void main () 
{
    colour = vec4(1.0,1.0,1.0,1.0) ;

    // show records with time less than cut, so can scan the cut upwards to see history 
    // float t = rpos.w * TimeDomain.y ; 
    // float w = Param.w > t ? 1. : 0. ;   
    // gl_Position = ISNormModelViewProjection * vec4 (vec3(rpos), w );
    // gl_Position = ISNormModelViewProjection * vec4( vec3(rpos), rpos.w*TimeDomain.y );

    // pass thru to geom.glsl
    flq = rflq ; 
    gl_Position = rpos ; 
    gl_PointSize = 1.0;

}


