#version 400

layout(location = 0) in vec4  rpos;
layout(location = 1) in ivec4 rflg;  

void main () 
{

    // pass thru to geom.glsl
    gl_Position = rpos ; 
    gl_PointSize = 1.0;

}


