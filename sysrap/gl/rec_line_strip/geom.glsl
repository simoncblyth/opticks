#version 410 core

// based on oglrap/gl/altrec/geom.glsl

uniform mat4 ModelViewProjection ;
uniform vec4 Param ;

layout (lines) in;
layout (line_strip, max_vertices = 2) out;

out vec4 fcolor ;


void main ()
{
    vec4 p0 = gl_in[0].gl_Position ;
    vec4 p1 = gl_in[1].gl_Position ;
    // two consequtive record positions with propagation times in .w

    float tc = Param.w  ;

    uint valid  = (uint(p0.w > 0.) << 0) + (uint(p1.w > 0.) << 1) + (uint(p1.w > p0.w) << 2) ;
    // valid : times > 0. and ordered correctly (no consideration of tc, input time Param.w)
    // permitting zero causes "future kink" issue

    uint select = (uint(tc > p0.w ) << 0) + (uint(tc < p1.w) << 1) + (0x1 << 2 )  ;
    // select : input time Param tc is between the two point times

    uint valid_select = valid & select ;
    // bitwise combination

    fcolor = vec4(1.0,1.0,1.0,1.0) ; ;

    if(valid_select == 0x7) // both points valid and with tc inbetween the points, so can mix to get position
    {
        gl_Position = ModelViewProjection * vec4(vec3(p0), 1.0) ;
        EmitVertex();

        vec3 pt = mix( vec3(p0), vec3(p1), (tc - p0.w)/(p1.w - p0.w) );
        gl_Position = ModelViewProjection * vec4( pt, 1.0 ) ;
        EmitVertex();
    }
    else if( valid == 0x7 && select == 0x5 )  // both points valid, but time is beyond them both
    {
        gl_Position = ModelViewProjection * vec4( vec3(p0), 1.0 ) ;
        EmitVertex();

        gl_Position = ModelViewProjection * vec4( vec3(p1), 1.0 ) ;
        EmitVertex();
    }
    EndPrimitive();   // MAYBE DO THIS INSIDE THE BRACKET ? 
}


