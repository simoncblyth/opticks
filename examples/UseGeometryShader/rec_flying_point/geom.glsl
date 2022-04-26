#version 410 core

uniform mat4 ModelViewProjection ;
uniform vec4 Param ; 

layout (lines) in;
layout (points, max_vertices = 1) out;

out vec4 fcolor ; 

void main () 
{
    vec4 p0 = gl_in[0].gl_Position  ;
    vec4 p1 = gl_in[1].gl_Position  ;
    float tc = Param.w  ;

    uint valid  = (uint(p0.w >= 0.)  << 0) + (uint(p1.w >= 0.) << 1) + (uint(p1.w > p0.w) << 2) ; 
    uint select = (uint(tc > p0.w ) << 0) + (uint(tc < p1.w) << 1) + (0x1 << 2 )  ;  
    uint vselect = valid & select ; 

    fcolor = vec4(1.0,1.0,1.0,1.0) ; ;

    if(vselect == 0x7) // both valid and straddling tc 
    {
        vec3 pt = mix( vec3(p0), vec3(p1), (tc - p0.w)/(p1.w - p0.w) );
        gl_Position = ModelViewProjection * vec4( pt, 1.0 ) ; 
        gl_PointSize = 2. ; 
        EmitVertex();
        EndPrimitive();
    }
    else if( valid == 0x7 && select == 0x5 )     // both valid and prior to tc 
    {
        vec3 pt = vec3(p1) ;
        gl_Position = ModelViewProjection * vec4( pt, 1.0 ) ; 
        gl_PointSize = 2. ; 
        EmitVertex();
        EndPrimitive();
    }
} 


