#version 410 core

layout(location = 0) in vec4  rpos;

void main () 
{
    gl_Position = rpos ; 
    gl_PointSize = 10.0;
}

