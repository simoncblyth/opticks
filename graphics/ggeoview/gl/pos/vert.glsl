#version 400

uniform mat4 ModelViewProjection ;
uniform mat4 ModelView ;

layout(location = 10) in vec3 vpos;
out vec3 colour;

void main () 
{
    colour = vec3(0.0,1.0,0.0) ;
    gl_Position = ModelViewProjection * vec4 (vpos, 1.0);
}


