#version 410 core

// https://www.opengl.org/wiki/Geometry_Shader

uniform mat4 ModelViewProjection ;
uniform vec4 Param ;
in vec3 deltapos[];

layout (points) in;
layout (line_strip, max_vertices = 2) out;

out vec4 fcolor ;


void main ()
{
    vec4 p0 = gl_in[0].gl_Position ;
    vec4 p1 = gl_in[0].gl_Position + vec4(deltapos[0], 0.) ;

    fcolor = vec4(1.0,1.0,1.0,1.0) ;

    gl_Position = ModelViewProjection * p0 ;
    EmitVertex();

    gl_Position = ModelViewProjection * p1  ;
    EmitVertex();

    EndPrimitive();
}
