#version 400

// axis passthrough to geometry shader

uniform mat4 ModelViewProjection ;
uniform mat4 ModelView ;
uniform vec4 LightPosition ; 
uniform vec4 Param ;


layout(location = 0) in vec4 vpos ;
layout(location = 1) in vec4 vdir ;

out vec3 direction ; 
out vec3 colour ; 

void main () 
{
    colour = vec3(1.0) ;

    direction = vdir.xyz ;

    //gl_Position = vec4( vec3(vpos) , 1.0);

    //gl_Position = vec4( vec3( ModelView * LightPosition ) , 1.0);

    gl_Position = vec4( vec3( LightPosition ) , 1.0);
}


