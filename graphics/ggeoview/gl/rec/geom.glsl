#version 400

// https://www.opengl.org/wiki/Geometry_Shader
// http://www.informit.com/articles/article.aspx?p=2120983&seqNum=2

uniform mat4 ISNormModelViewProjection ;
uniform vec4 TimeDomain ;
uniform vec4 ColorDomain ;
uniform vec4 Param ; 
uniform ivec4 Pick ;
uniform ivec4 RecSelect ; 

uniform sampler1D Colors ;

in vec4 colour[];
in uvec4 flq[];
in ivec4 sel[];

layout (lines) in;

layout (points, max_vertices = 1) out;
out vec4 fcolour ; 


// NB this is called against an "un-partitioned" record array
//    ie even getting records crossing between photons
//    so dropping invalids is the first thing to do

void main () 
{
    uint seqhis = sel[0].x ; 
    uint seqmat = sel[0].y ; 
    if( RecSelect.x > 0 && RecSelect.x != seqhis )  return ;
    if( RecSelect.y > 0 && RecSelect.y != seqmat )  return ;


    vec4 p0 = gl_in[0].gl_Position  ;
    vec4 p1 = gl_in[1].gl_Position  ;
    float tc = Param.w / TimeDomain.y ;

    uint valid  = (uint(p0.w > 0.)  << 0) + (uint(p1.w > 0.) << 1) + (uint(p1.w > p0.w) << 2) ; 
    uint select = (uint(tc > p0.w ) << 0) + (uint(tc < p1.w) << 1) + (uint(Pick.w == 0 || gl_PrimitiveIDIn/10 == Pick.w) << 2) ;  
    uint vselect = valid & select ; 

    //
    // material coloring  
    //
    //  float idxcol = (float(flq[1].x) - 1.0 + 0.5)/ColorDomain.y ;            
    //
    //     flq[0].x  sorta working m1? colors 
    //     flq[0].y  sorta working m2?
    //     flq[0].z  all red, expected to be zero (yielding -1 off edge of texture, so picks up value of 1 based "1")
    //     flq[0].w  variable colors, looks consistent with flags
    //
    //     flq[1].x  a bit more distinct m1  
    //     flq[1].y
    //     flq[1].z  
    //     flq[1].w  not much variation  
    //
    //
    //  history coloring
    //
       float idxcol  = (32.0 + float(flq[1].w) - 1.0 + 0.5)/ColorDomain.y ;    
    //
    //     flq[0].x   color variation
    //     flq[0].y   color variation
    //     flq[0].z   all dull grey, consistent with expected zero yielding "-1" and landing on buffer prefill 0x444444 
    //     flq[0].w   very subtle coloring mostly along muon path,  
    //
    //     flq[1].x   more distinct color variation
    //     flq[1].y   more distinct color variation
    //     flq[1].z   all dull grey (buffer prefill) 
    //     flq[1].w   more obvious, but still too much white : off by one maybe?
    // 
    //  problem is that the most common step flag is BT:boundary transmit (now cyan)
    //  which makes flying point view not so obvious  
    //

    if(vselect == 0x7) // both valid and straddling tc 
    {
        vec3 pt = mix( vec3(p0), vec3(p1), (tc - p0.w)/(p1.w - p0.w) );
        gl_Position = ISNormModelViewProjection * vec4( pt, 1.0 ) ; 
        fcolour = texture(Colors, idxcol   ) ;
        EmitVertex();
        EndPrimitive();
    }
    else if( valid == 0x7 && select == 0x5 ) // both valid and prior to tc 
    {
        gl_Position = ISNormModelViewProjection * vec4( vec3(p1), 1.0 ) ; 
        fcolour = texture(Colors, idxcol ) ; 
        EmitVertex();
        EndPrimitive();
    }

} 



/*


+In [2]: np.linspace(0.,1.,5+1 )
+Out[2]: array([ 0. ,  0.2,  0.4,  0.6,  0.8,  1. ])
+                RRRRRRRRGGGGGGMMMMMM
+ 
+ So to pick via an integer would need 
+
+In [10]: (np.arange(0,5) + 0.5)/5.              (float(i) + 0.5)/5.0  lands mid-bin
+Out[10]: array([ 0.1,  0.3,  0.5,  0.7,  0.9])
+


*/
