#version 400
//
//  Below ascii art shows expected pattern of slots and times for MAXREC 5 
//
//  * remember that from point of view of shader the input time is **CONSTANT**
//    think of the drawing as a chart plotter tracing over all the steps of all the photons, 
//    this shader determines when to put the pen down onto the paper
//    
//    * it needs to lift pen between photons and avoid invalids 
//
//  * slot indices are presented modulo 5
//  * negative times indicates unset
//  * dt < 0. indicates p1 invalid
//  
//
//     |                                          
//     |                                           
//     t                                            
//     |          3                                  
//     |                                          4
//     |      2                                3
//     |    1                               2              
//     |  0                   2          1              1 
//     |                    1         0               0          0
//     +-----------------0--------> slot ------------------------------------->
//     |                                     
//     |              4         3 4                        2 3 4    1 2 3 4 
//     |
//
//   
//  
//  * geom shader gets to see all contiguous pairs 
//    (including invalid pairs that cross between different photons)
//
//  * shader uses one input time cut Param.w to provide history scrubbing 
//
//  * a pair of contiguous recs corresponding to a potential line
//
//
//  Choices over what to do with the pair:
//
//  * do nothing with this pair, eg for invalids 
//  * interpolate the positions to find an intermediate position 
//    as a function of input time 
//
//  * throw away one position, retaining the other 
//  
// https://www.opengl.org/wiki/Geometry_Shader
// http://www.informit.com/articles/article.aspx?p=2120983&seqNum=2

#incl dynamic.h

uniform mat4 ISNormModelViewProjection ;
uniform vec4 TimeDomain ;
uniform vec4 Param ; 
uniform ivec4 Selection ;
uniform ivec4 Pick ;
uniform ivec4 RecSelect ; 

in ivec4 sel[];
in vec4 polarization[];
layout (lines) in;
layout (line_strip, max_vertices = 2) out;
out vec4 fcolour ; 


void main () 
{
    uint seqhis = sel[0].x ; 
    uint seqmat = sel[0].y ; 
    if( RecSelect.x > 0 && RecSelect.x != seqhis )  return ;
    if( RecSelect.y > 0 && RecSelect.y != seqmat )  return ;

    uint photon_id = gl_PrimitiveIDIn/MAXREC ;                 // https://www.opengl.org/sdk/docs/man/html/gl_PrimitiveIDIn.xhtml

    vec4 p0 = gl_in[0].gl_Position  ;
    vec4 p1 = gl_in[1].gl_Position  ;
    float tc = Param.w / TimeDomain.y ;  // as time comparisons done before un-snorming 

    uint valid  = (uint(p0.w > 0.)  << 0) + (uint(p1.w > 0.) << 1) + (uint(p1.w > p0.w) << 2) ; 

  //uint select = (uint(tc > p0.w ) << 0) + (uint(tc < p1.w) << 1) + (uint(Pick.w == 0 || photon_id == Pick.w) << 2) ;  
    uint select = (uint(tc > p0.w ) << 0) + (uint(tc < p1.w) << 1) + (uint(Pick.x == 0 || photon_id % Pick.x == 0) << 2) ;  

    uint vselect = valid & select ; 

    if(vselect == 0x7) // both valid and straddling tc 
    {
        gl_Position = ISNormModelViewProjection * vec4(vec3(p0), 1.0) ; 
        fcolour = vec4(vec3(polarization[0]), 1.0) ;
        EmitVertex();

        vec3 pt = mix( vec3(p0), vec3(p1), (tc - p0.w)/(p1.w - p0.w) ); 
        gl_Position = ISNormModelViewProjection * vec4( pt, 1.0 ) ; 
        fcolour = vec4(vec3(polarization[1]), 1.0) ;
        EmitVertex();

        EndPrimitive();
    }
    else if( valid == 0x7 && select == 0x5 ) // both valid and prior to tc 
    {
        gl_Position = ISNormModelViewProjection * vec4(vec3(p0), 1.0) ; 
        fcolour = vec4(vec3(polarization[0]), 1.0) ;
        EmitVertex();

        gl_Position = ISNormModelViewProjection * vec4(vec3(p1), 1.0) ; 
        fcolour = vec4(vec3(polarization[1]), 1.0) ;
        EmitVertex();

        EndPrimitive();
    }

    //
    //  Cannot form a line with only one valid point ? unless conjure a constant direction.
    //  The only hope is that a prior "thread" got the valid point as
    //  the second of a pair. 
    //  Perhaps that means must draw with GL_LINE_STRIP rather than GL_LINES in order
    //  that the geometry shader sees each vertex twice (?)   YES : SEEMS SO
    //  
    //  Hmm how to select single photons/steps ?  
    //  
    //  * Storing photon identifies occupies ~22 bits at least (1 << 22)/1e6 ~ 4.19
    //  * Step identifiers 
    //
    //  * https://www.opengl.org/wiki/Built-in_Variable_(GLSL) 
    //
    //  * https://www.opengl.org/sdk/docs/man/html/gl_VertexID.xhtml
    // 
    //    non-indexed: it is the effective index of the current vertex (number of vertices processed + *first* value)
    //    indexed:   index used to fetch this vertex from the buffer
    //
    //  * control the the glDrawArrays first/count to pick the desired range.
    //
    //  * adopt glDrawElements and control the indices
    //

} 
