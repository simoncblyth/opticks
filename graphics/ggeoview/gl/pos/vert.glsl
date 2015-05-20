#version 400

uniform mat4 ModelViewProjection ;
uniform mat4 ModelView ;

layout(location = 0) in vec3  vpos;
layout(location = 3) in ivec4 iflg;

out vec3 colour;

void main () 
{
    colour = vec3(1.0,1.0,1.0) ;

    if(     iflg.x == 11) colour = vec3(1.0,0.0,0.0) ;
    else if(iflg.x == 17) colour = vec3(0.0,1.0,0.0) ;
    else if(iflg.x >  17) colour = vec3(0.0,0.0,1.0) ;

    gl_Position = ModelViewProjection * vec4 (vpos, 1.0);

    gl_PointSize = 1.0;

}


