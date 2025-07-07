#version 410 core

layout(location = 0) in vec4 rpos ;
layout(location = 1) in vec4 rdel ;

out vec3 deltapos ;

void main ()
{
    deltapos = rdel.xyz ;
    gl_Position = vec4( rpos.xyz, 1.0);
}


