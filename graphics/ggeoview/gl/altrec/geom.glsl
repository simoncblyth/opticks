#version 400
//
//  Below ascii art shows expected pattern of slots and times for MAXREC 5 
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

uniform mat4 ISNormModelViewProjection ;
uniform vec4 TimeDomain ;
uniform vec4 Param ; 

layout (lines) in;

layout (line_strip, max_vertices = 2) out;
//layout (points, max_vertices = 1) out;

out vec4 fcolour ; 


void main () 
{
    vec4 p0 = gl_in[0].gl_Position  ;
    vec4 p1 = gl_in[1].gl_Position  ;
    float tc = Param.w / TimeDomain.y ;  // as time comparisons done before un-snorming 
    fcolour = vec4(0.0,1.0,1.0,1.0) ;

    uint valid  = (uint(p0.w > 0.)  << 0) + (uint(p1.w > 0.) << 1) + (uint(p1.w > p0.w) << 2) ; 
    uint select = (uint(tc > p0.w ) << 0) + (uint(tc < p1.w) << 1) + (1 << 2) ;
    uint vselect = valid & select ; 

    if(vselect == 0x7) // both valid and straddling tc 
    {
        gl_Position = ISNormModelViewProjection * vec4(vec3(p0), 1.0) ; 
        EmitVertex();

        vec3 pt = mix( vec3(p0), vec3(p1), (tc - p0.w)/(p1.w - p0.w) ); 
        gl_Position = ISNormModelViewProjection * vec4( pt, 1.0 ) ; 
        EmitVertex();

        EndPrimitive();
    }
    // hmm cannot form a line with only one valid point ?
   


} 
