#version 410 core

layout(location = 0) in vec4 rpos;
layout(location = 1) in vec4 rpol;

out vec4 polarization ;

void main ()
{
    polarization = rpol ;
    gl_Position = rpos ;
    gl_PointSize = 10.0;
}

