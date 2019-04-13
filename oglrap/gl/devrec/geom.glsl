#version 410 core
// devrec/geom.glsl    shortline or vector presentation, control length with param.y

#incl dynamic.h

uniform mat4 ISNormModelViewProjection ;
uniform vec4 TimeDomain ;
uniform vec4 ColorDomain ;
uniform vec4 Param ; 
uniform ivec4 Selection ;
uniform ivec4 Pick ;
uniform ivec4 RecSelect ; 
uniform ivec4 ColorParam ;
uniform ivec4 PickPhoton ;

uniform sampler1D Colors ;

in ivec4 sel[];
in vec4 polarization[];
in uvec4 flags[];
in uvec4 flq[];

layout (lines) in;
layout (line_strip, max_vertices = 2) out;

out vec4 fcolor ; 

void main () 
{
    uint seqhis = sel[0].x ; 
    uint seqmat = sel[0].y ; 
    if( RecSelect.x > 0 && RecSelect.x != seqhis )  return ;
    if( RecSelect.y > 0 && RecSelect.y != seqmat )  return ;

    uint photon_id = gl_PrimitiveIDIn/MAXREC ;   
    if( PickPhoton.x > 0 && PickPhoton.y > 0 && PickPhoton.x != photon_id )  return ;


    vec4 p0 = gl_in[0].gl_Position  ;
    vec4 p1 = gl_in[1].gl_Position  ;
    float tc = Param.w / TimeDomain.y ;  // as time comparisons done before un-snorming 
    //float ns = Param.y/TimeDomain.y ; 
    float dfrac = Param.y ;   //  vector length control 


    uint valid  = (uint(p0.w > 0.)  << 0) + (uint(p1.w > 0.) << 1) + (uint(p1.w > p0.w) << 2) ; 
    uint select = (uint(tc > p0.w ) << 0) + (uint(tc < p1.w) << 1) + (uint(Pick.x == 0 || photon_id % Pick.x == 0) << 2) ;  

    uint vselect = valid & select ; 

#incl fcolor.h

    if(vselect == 0x7) // both valid and straddling tc 
    {
        float frac = (tc - p0.w)/(p1.w - p0.w)  ;
        vec3 pt0 = mix( vec3(p0), vec3(p1), frac ); 
        gl_Position = ISNormModelViewProjection * vec4( pt0, 1.0 ) ; 


        EmitVertex();

        vec3 pt1 = mix( vec3(p0), vec3(p1), frac + dfrac ); 
        gl_Position = ISNormModelViewProjection * vec4( pt1, 1.0 ) ; 
        EmitVertex();

        EndPrimitive();
    }
    else if( valid == 0x7 && select == 0x5 ) // both valid and prior to tc 
    {
        gl_Position = ISNormModelViewProjection * vec4( vec3(p1), 1.0 ) ; 
        EmitVertex();

        gl_Position = ISNormModelViewProjection * vec4( vec3(p1 + dfrac*(p1-p0)) , 1.0 ) ; 
        EmitVertex();

        EndPrimitive();
    }
} 

