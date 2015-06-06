#version 400
#enum photon_flags.h

uniform mat4 ModelViewProjection ;
uniform mat4 ModelView ;
uniform ivec4 Selection ; 
uniform ivec4 Flags ; 

layout(location = 0) in vec3  vpos;
layout(location = 2) in vec3  vpol;
layout(location = 3) in ivec4 iflg;  

// see tail of generate.cu 

out vec4 colour;

void main () 
{
    float w = 1.0 ; 
    colour = vec4(0.5,0.5,0.5,1.0) ;

    // NB the comparison is using 1-based cos_theta signed boundary codes, 0 means miss
    //
    if(     iflg.x == Selection.x ) colour = vec4(1.0,1.0,1.0,1.0) ;
    else if(iflg.x == Selection.y ) colour = vec4(1.0,0.0,0.0,1.0) ;
    else if(iflg.x == Selection.z ) colour = vec4(0.0,1.0,0.0,1.0) ;
    else if(iflg.x == Selection.w ) colour = vec4(0.0,0.0,1.0,1.0) ;
    else if((iflg.w & Flags.x) != 0 ) colour = vec4(1.0,0.0,1.0,1.0) ;
    else                      
    {
        colour = vec4(0.0,0.0,0.0,0.0) ; 
        w = 0.0 ;      // scoot unselected points off to infinity and beyond
    }

    gl_Position = ModelViewProjection * vec4 (vpos, w );
    gl_PointSize = 1.0;

}


