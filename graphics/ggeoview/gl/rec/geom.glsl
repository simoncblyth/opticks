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

    float idxcol = (flq[0].y + 0.5)/5. ;


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
