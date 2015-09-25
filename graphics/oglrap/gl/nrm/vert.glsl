#version 400

uniform mat4 ModelViewProjection ;
uniform mat4 ModelView ;
uniform vec4 ClipPlane ;
uniform vec4 LightPosition ; 
uniform vec4 Param ;
uniform ivec4 NrmParam ;

uniform vec4 ColorDomain ;
uniform sampler1D Colors ;


layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_colour;
layout(location = 2) in vec3 vertex_normal;

float gl_ClipDistance[1]; 

out vec4 colour;

void main () 
{
    //
    // NB using flipped normal, for lighting from inside geometry 
    //
    //    normals are expected to be outwards so the natural 
    //    sign of costheta is negative when the light is inside geometry 
    //    thus in order to see something flip the normals 
    //

    float flip = NrmParam.x == 1 ? -1. : 1. ;

    vec3 normal = flip * normalize(vec3( ModelView * vec4 (vertex_normal, 0.0)));

    vec3 vpos_e = vec3( ModelView * vec4 (vertex_position, 1.0));  // vertex position in eye space 

    gl_ClipDistance[0] = dot(vec4(vertex_position, 1.0), ClipPlane);

    vec3 ambient = vec3(0.1, 0.1, 0.1) ;

#incl colour.h

    gl_Position = ModelViewProjection * vec4 (vertex_position, 1.0);

}


