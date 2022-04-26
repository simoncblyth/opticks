#version 410 core

uniform mat4 ModelViewProjection ;

layout(location = 0) in vec4  vpos;
layout(location = 1) in vec4  vdir;
layout(location = 2) in vec4  vcol;  

out vec4 colour;

void main ()  
{
    colour = vec4(0.5,0.5,0.5,1.0) ;
    gl_Position = ModelViewProjection * vpos ;
    gl_PointSize = 5.0;

}

