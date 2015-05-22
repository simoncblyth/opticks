#version 400

uniform mat4 ModelViewProjection ;
uniform mat4 ModelView ;
uniform ivec4 Selection ; 

layout(location = 0) in vec3  vpos;
layout(location = 2) in vec3  vpol;
layout(location = 3) in ivec4 iflg;  

// see tail of generate.cu 

out vec3 colour;

void main () 
{
    colour = vec3(0.5,0.5,0.5) ;

    // NB the comparison is using 1-based cos_theta signed boundary codes, 0 means miss
    //
    if(     iflg.x == Selection.x ) colour = vec3(1.0,0.0,0.0) ;
    else if(iflg.x == Selection.y ) colour = vec3(0.0,1.0,0.0) ;
    else if(iflg.x == Selection.z ) colour = vec3(0.0,0.0,1.0) ;
    else if(iflg.x == Selection.w ) colour = vec3(1.0,1.0,1.0) ;
    else                            colour = vec3(0.5,0.5,0.5) ;


    gl_Position = ModelViewProjection * vec4 (vpos, 1.0);

    gl_PointSize = 1.0;

}


