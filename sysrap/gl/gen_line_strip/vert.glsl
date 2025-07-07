#version 410 core

layout(location = 0) in vec4 rpos ;
layout(location = 1) in vec4 rdel ;

out vec4 deltapos ;

void main ()
{
    deltapos = rdel ;
    gl_Position = rpos ;
    gl_PointSize = 10.0;
}


