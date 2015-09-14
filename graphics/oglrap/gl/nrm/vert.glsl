#version 400

uniform mat4 ModelViewProjection ;
uniform mat4 ModelView ;
uniform vec4 ClipPlane ;
uniform vec4 LightPosition ; 
uniform vec4 Param ;
uniform ivec4 NrmParam ;


layout(location = 0) in vec3 vertex_position;
layout(location = 1) in vec3 vertex_colour;
layout(location = 2) in vec3 vertex_normal;
//layout(location = 3) in vec2 vertex_texcoord;

float gl_ClipDistance[1]; 

out vec4 colour;
//out vec2 texcoord;

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

    vec3 lpos_e = vec3( ModelView * LightPosition );   

    vec3 vpos_e = vec3( ModelView * vec4 (vertex_position, 1.0));  // vertex position in eye space 

    vec3 ldir_e = normalize(lpos_e - vpos_e);

    float diffuse = clamp( dot(normal, ldir_e), 0, 1) ;

    //float diffuse =  abs(dot(normal, ldir_e)) ;   // absolution rather than clamping makes a big difference  

    gl_ClipDistance[0] = dot(vec4(vertex_position, 1.0), ClipPlane);

    //colour = vec4( normal*0.5 + 0.5, 1.0 - Param.z ) ;    // 1 - alpha 
    //colour = vec4( vertex_colour, 1.0 - Param.z ); 

    vec3 ambient = vec3(0.1, 0.1, 0.1) ;

    colour = vec4( ambient + vec3(diffuse) * vertex_colour , 1.0 - Param.z );


    gl_Position = ModelViewProjection * vec4 (vertex_position, 1.0);

    //texcoord = vertex_texcoord;

}


