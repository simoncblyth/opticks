#version 400
// devrec , see altrec for explanation of record structure

#incl dynamic.h
#incl define.h

uniform mat4 ISNormModelViewProjection ;
uniform vec4 TimeDomain ;
uniform vec4 ColorDomain ;
uniform vec4 Param ; 
uniform ivec4 Selection ;
uniform ivec4 Pick ;
uniform ivec4 RecSelect ; 

uniform sampler1D Colors ;


in ivec4 sel[];
in vec4 polarization[];
in uvec4 flags[];
in uvec4 flq[];


layout (lines) in;
layout (line_strip, max_vertices = 2) out;

out vec4 fcolour ; 

//
// Correspondence to OptiX qtys
//
//   flags[0].x  <=>  polw.ushort_.z    boundary / m1
//   flags[0].y  <=>  polw.ushort_.w    16 bits of history 
//
// For example did IDENTIY_CHECK with 
// 
//   uint verify_photon_id = flags[0].x | flags[0].y << 16 ;  
//
// TODO: 
//
// * try draw arrays approach to picking : maybe much more efficient 
// * rejig flags and provide gui for selection based on them
//
// * will (p1.w - p0.w) <= 0  identify all invalids ? 
//
//   * two ways to be invalid : UNSET and ABUTTING
//

void main () 
{
    uint seqhis = sel[0].x ; 
    uint seqmat = sel[0].y ; 
    if( RecSelect.x > 0 && RecSelect.x != seqhis )  return ;
    if( RecSelect.y > 0 && RecSelect.y != seqmat )  return ;

    // m1 coloring  
    //float idxcol0 = (float(flq[0].x) - 1.0 + 0.5)/ColorDomain.y ;            
    //float idxcol1 = (float(flq[1].x) - 1.0 + 0.5)/ColorDomain.y ;            

    // flag coloring  
    float idxcol0  = (32.0 + float(flq[0].w) - 1.0 + 0.5)/ColorDomain.y ;    
    float idxcol1  = (32.0 + float(flq[1].w) - 1.0 + 0.5)/ColorDomain.y ;    



    vec4 p0 = gl_in[0].gl_Position  ;
    vec4 p1 = gl_in[1].gl_Position  ;
    float tc = Param.w / TimeDomain.y ;  // as time comparisons done before un-snorming 
    float ns = 1.0/TimeDomain.y ; 

    uint photon_id = gl_PrimitiveIDIn/MAXREC ;                 // https://www.opengl.org/sdk/docs/man/html/gl_PrimitiveIDIn.xhtml

    uint valid  = (uint(p0.w > 0.)  << 0) + (uint(p1.w > 0.) << 1) + (uint(p1.w > p0.w) << 2) ; 

  //uint select = (uint(tc > p0.w ) << 0) + (uint(tc < p1.w) << 1) + (uint(flags[0].x == Pick.y) << 2 ) ;
  //uint select = (uint(tc > p0.w ) << 0) + (uint(tc < p1.w) << 1) + (uint(Pick.w == 0 || photon_id == Pick.w) << 2) ;  
    uint select = (uint(tc > p0.w ) << 0) + (uint(tc < p1.w) << 1) + (uint(Pick.x == 0 || photon_id % Pick.x == 0) << 2) ;  

    uint vselect = valid & select ; 



    if(vselect == 0x7) // both valid and straddling tc 
    {
        vec3 pt0 = mix( vec3(p0), vec3(p1), (tc - p0.w)/(p1.w - p0.w) ); 
        gl_Position = ISNormModelViewProjection * vec4( pt0, 1.0 ) ; 

        //fcolour = vec4(vec3(polarization[1]), 1.0) ;
        fcolour = texture(Colors, idxcol0   ) ;

        EmitVertex();

        vec3 pt1 = mix( vec3(p0), vec3(p1), (tc + ns - p0.w)/(p1.w - p0.w) ); 
        gl_Position = ISNormModelViewProjection * vec4( pt1, 1.0 ) ; 

        //fcolour = vec4(vec3(polarization[1]), 1.0) ;
        fcolour = texture(Colors, idxcol1   ) ;

        EmitVertex();

        EndPrimitive();

    }
    else if( valid == 0x7 && select == 0x5 ) // both valid and prior to tc 
    {
        gl_Position = ISNormModelViewProjection * vec4( vec3(p1), 1.0 ) ; 
        //fcolour = vec4(vec3(polarization[1]), 1.0) ;
        fcolour = texture(Colors, idxcol1   ) ;
        EmitVertex();

        gl_Position = ISNormModelViewProjection * vec4( vec3(p1 + 0.1*(p1-p0)) , 1.0 ) ; 
        //fcolour = vec4(vec3(polarization[1]), 1.0) ;
        fcolour = texture(Colors, idxcol1   ) ;
        EmitVertex();

        EndPrimitive();

    }


/*
    if(vselect == 0x7) // both valid and straddling tc 
    {
        gl_Position = ISNormModelViewProjection * vec4(vec3(p0), 1.0) ; 
        fcolour = vec4(vec3(polarization[0]+1.f)/2.f, 1.0) ;
        EmitVertex();

        vec3 pt = mix( vec3(p0), vec3(p1), (tc - p0.w)/(p1.w - p0.w) ); 
        gl_Position = ISNormModelViewProjection * vec4( pt, 1.0 ) ; 
        fcolour = vec4(vec3(polarization[1]+1.f)/2.f, 1.0) ;
        EmitVertex();

        EndPrimitive();
    }

    else if( valid == 0x7 && select == 0x5 ) // both valid and prior to tc 
    {
        gl_Position = ISNormModelViewProjection * vec4(vec3(p0), 1.0) ; 
        fcolour = vec4(vec3(polarization[0]+1.f)/2.f, 1.0) ;
        EmitVertex();

        gl_Position = ISNormModelViewProjection * vec4(vec3(p1), 1.0) ; 
        fcolour = vec4(vec3(polarization[1]+1.f)/2.f, 1.0) ;
        EmitVertex();

        EndPrimitive();
    }
*/


} 

