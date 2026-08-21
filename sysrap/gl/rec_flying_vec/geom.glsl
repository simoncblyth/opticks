#version 410 core

uniform mat4 ModelViewProjection ;
uniform vec4 Param ;
uniform vec4 Auxil ;

in vec4 polarization[];

layout (lines) in;
layout (line_strip, max_vertices = 2) out;

out vec4 fcolor ;

#include "../util/wavelengthToSRGB.glsl"

//#define PERSIST 1
// * With PERSIST defined the old rings remain visible after time has gone by
// * Without PERSIST get the traditional flying point visualization

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

    //fcolor = vec4(1.0,1.0,1.0,1.0) ;
    //fcolor = vec4(vec3(polarization[0]), 1.0);

    vec3 rgb = wavelengthToSRGB( polarization[0].w );  // sphoton.h polarization[0].w is wavelength_nm
    fcolor = vec4(vec3(rgb), 1.0);

    if(valid_select == 0x7) // both points valid and with tc inbetween the points, so can mix to get position
    {
        float frac0 = (tc - p0.w)/(p1.w - p0.w);
        vec3 pt0 = mix( vec3(p0), vec3(p1), frac0 );
        gl_Position = ModelViewProjection * vec4( pt0, 1.0 ) ;
        gl_PointSize = 2. ;
        EmitVertex();

        float frac1 = (Auxil.x + tc - p0.w)/(p1.w - p0.w);
        vec3 pt1 = mix( vec3(p0), vec3(p1), frac1 );
        gl_Position = ModelViewProjection * vec4( pt1, 1.0 ) ;
        gl_PointSize = 2. ;
        EmitVertex();

        EndPrimitive();
    }
#ifdef PERSIST
    else if( valid == 0x7 && select == 0x5 )  // both points valid, but time is beyond them both
    {
        // First vertex at p1
        vec3 pt0 = vec3(p1) ;
        gl_Position = ModelViewProjection * vec4( pt0, 1.0 ) ;
        gl_PointSize = 2. ;
        EmitVertex();

        // Second vertex extrapolates 0.1 ns past p1 (fraction = 1.0 + 0.1 / dt)
        float frac1 = 1.0 + (Auxil.x / (p1.w - p0.w));
        vec3 pt1 = mix( vec3(p0), vec3(p1), frac1 );
        gl_Position = ModelViewProjection * vec4( pt1, 1.0 );
        gl_PointSize = 2.0;
        EmitVertex();

        EndPrimitive();
    }
#endif
}


