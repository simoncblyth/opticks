#version 400

uniform mat4 ModelViewProjection ;
uniform mat4 CenterExtentUnNormalize ;
uniform mat4 ModelView ;

layout(location = 0) in vec4  rpos;
layout(location = 1) in ivec4 rflg;  


out vec4 colour;

void main () 
{
    colour = vec4(0.5,0.5,0.5,1.0) ;
    //gl_Position = ModelViewProjection * vec4 (vec3(rpos), 1.0 );
    gl_Position = CenterExtentUnNormalize * ModelViewProjection * vec4 (vec3(rpos), 1.0 );
    gl_PointSize = 1.0;

}


