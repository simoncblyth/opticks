#version 410 core

// https://www.opengl.org/wiki/Geometry_Shader

uniform mat4 ModelViewProjection ;
uniform vec4 Param ;
in vec4 deltapos[];

layout (points) in;
layout (line_strip, max_vertices = 2) out;

out vec4 fcolor ;


void main ()
{
    vec4 p0 = gl_in[0].gl_Position ;
    vec4 p1 = gl_in[0].gl_Position + deltapos[0]  ;
    // p1.w is gibberish : adding time to step_length

    float t0 = p0.w ;
    float tc = Param.w  ;

    fcolor = vec4(1.0,1.0,1.0,1.0) ;

    if( tc > t0 )
    {
        gl_Position = ModelViewProjection * vec4( vec3(p0), 1.0 ) ;
        EmitVertex();

        gl_Position = ModelViewProjection * vec4( vec3(p1), 1.0 ) ;
        EmitVertex();
    }

    EndPrimitive();
}
