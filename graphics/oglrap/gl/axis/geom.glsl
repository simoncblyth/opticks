#version 400


uniform mat4 Projection ;
uniform mat4 ModelViewProjection ;
uniform vec4 Param ; 

in vec3 direction[];
in vec3 colour[];

layout (points) in;
layout (line_strip, max_vertices = 2) out;

out vec3 fcolour ; 


void main () 
{
    gl_Position = ModelViewProjection * gl_in[0].gl_Position ;
    fcolour = colour[0] ;
    EmitVertex();

    gl_Position = ModelViewProjection * ( gl_in[0].gl_Position + Param.x*vec4(direction[0], 0.) ) ;
    fcolour = colour[0] ;
    EmitVertex();

    EndPrimitive();

} 
