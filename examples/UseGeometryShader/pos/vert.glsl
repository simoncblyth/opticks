#version 410 core

uniform mat4 ModelViewProjection ;

layout(location = 0) in vec4 rpos;
//layout(location = 1) in vec4  vdir;
//layout(location = 2) in vec4  vcol;  

out vec4 colour;

void main ()  
{
    vec4 pos = vec4(rpos) ; 
    pos.w = 1. ;  
    colour = vec4(1.0,1.0,1.0,1.0) ;
    gl_Position = ModelViewProjection * pos ;
    gl_PointSize = 5.0;
    //gl_PointSize = 100.0;

}

