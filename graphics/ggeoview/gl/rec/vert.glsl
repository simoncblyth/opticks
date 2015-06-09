#version 400

uniform mat4 ModelViewProjection ;
uniform mat4 ISNormModelViewProjection ;
uniform mat4 ModelView ;
uniform vec4 Param ;
uniform vec4 TimeDomain ;

layout(location = 0) in vec4  rpos;
layout(location = 1) in ivec4 rflg;  


out vec4 colour;

void main () 
{
    colour = vec4(0.5,0.5,0.5,1.0) ;

    float t = rpos.w * TimeDomain.y ; 

    float w = Param.w > t ? 1. : 0. ;   
    // show records with time less than cut, so can scan the cut upwards to see history 

    gl_Position = ISNormModelViewProjection * vec4 (vec3(rpos), w );

    gl_PointSize = 1.0;


}


