#version 410 core

uniform mat4 ModelViewProjection ;
uniform vec4 Param ; 
uniform vec4 post_center ; 
uniform vec4 post_extent ; 

layout (lines) in;
layout (points, max_vertices = 1) out;

out vec4 fcolor ; 

void main () 
{
    //vec4 p0 = gl_in[0].gl_Position*post_extent + post_center ;
    //vec4 p1 = gl_in[1].gl_Position*post_extent + post_center ;

    vec4 p0 = gl_in[0].gl_Position ; 
    vec4 p1 = gl_in[1].gl_Position ; 
    // two consequtive record positions with propagation times in .w  

    float tc = Param.w  ;

    uint valid  = (uint(p0.w >= 0.) << 0) + (uint(p1.w >= 0.) << 1) + (uint(p1.w > p0.w) << 2) ; 
    // valid : times >= 0. and ordered correctly (no consideration of tc, input time Param.w)

    uint select = (uint(tc > p0.w ) << 0) + (uint(tc < p1.w) << 1) + (0x1 << 2 )  ;
    // select : input time Param tc is between the two points  
  
    uint vselect = valid & select ;
    // bitwise combination  

    fcolor = vec4(1.0,1.0,1.0,1.0) ; ;

    if(vselect == 0x7) // both points valid and with tc inbetween the points, so can mix to get position 
    {
        vec3 pt = mix( vec3(p0), vec3(p1), (tc - p0.w)/(p1.w - p0.w) );
        gl_Position = ModelViewProjection * vec4( pt, 1.0 ) ; 
        gl_PointSize = 2. ; 
        EmitVertex();
        EndPrimitive();
    }
    //else if( valid == 0x7 && select == 0x5 )  // both points valid, but time is beyond them both 
    //{
    //    vec3 pt = vec3(p1) ;
    //    gl_Position = ModelViewProjection * vec4( pt, 1.0 ) ; 
    //    gl_PointSize = 2. ; 
    //    EmitVertex();
    //    EndPrimitive();
    //}
    // * Uncommenting the above block makes the time gone by 2nd point p1 persist beyond its life 
    // * With it commented get the more traditional flying point visualization 
} 


