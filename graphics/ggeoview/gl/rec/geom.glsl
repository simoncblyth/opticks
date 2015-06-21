#version 400

// https://www.opengl.org/wiki/Geometry_Shader
// http://www.informit.com/articles/article.aspx?p=2120983&seqNum=2

uniform mat4 ISNormModelViewProjection ;
uniform vec4 TimeDomain ;
uniform vec4 Param ; 
uniform sampler1D Colors ;

in vec4 colour[];
in ivec4 flq[];

layout (lines) in;

layout (points, max_vertices = 1) out;
out vec4 fcolour ; 

void main () 
{
    vec4 p0 = gl_in[0].gl_Position  ;
    vec4 p1 = gl_in[1].gl_Position  ;

    float dt = p1.w - p0.w ; 
    if(dt <= 0 ) return ;

    float tc = Param.w / TimeDomain.y ;

    uint m1 = flq[0].x ;

    // confirmed that the color texture is providing expected colors when artificially set m1 here
    //uint m1 = 12 ;    // palegreen
    //uint m1 = 13 ;    // yellow
    //uint m1 = 14 ;    // pink (but murky)
    //uint m1 = 15 ;    // hotpink
    //uint m1 = 1 ;     // azure looks indistuishable from white
    //uint m1 = 2 ;       // 

    float idxcol = (float(m1) - 1.0 + 0.5)/25. ;

    if( tc > p0.w && tc < p1.w  )     // tc between point times 
    {
        vec3 pt = mix( vec3(p0), vec3(p1), (tc - p0.w)/dt );
        gl_Position = ISNormModelViewProjection * vec4( pt, 1.0 ) ; 

        //fcolour = colour[0] ;
        fcolour = texture(Colors, idxcol   ) ;
        EmitVertex();
        EndPrimitive();
    }
    else if (tc > p0.w && tc > p1.w )
    {
        gl_Position = ISNormModelViewProjection * vec4( vec3(p1), 1.0 ) ; 

        //fcolour = colour[0] ;
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
