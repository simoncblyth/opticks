#version 400
// devrec , see altrec for explanation of record structure

uniform mat4 ISNormModelViewProjection ;
uniform vec4 TimeDomain ;
uniform vec4 Param ; 
uniform ivec4 Selection ;
uniform ivec4 Pick ;

in vec4 polarization[];
in ivec4 flags[];
layout (lines) in;
layout (line_strip, max_vertices = 2) out;
out vec4 fcolour ; 


void main () 
{
    vec4 p0 = gl_in[0].gl_Position  ;
    vec4 p1 = gl_in[1].gl_Position  ;
    float tc = Param.w / TimeDomain.y ;  // as time comparisons done before un-snorming 

    //fcolour = vec4(0.0,1.0,1.0,1.0) ;


   //  TODO: visible comparisons with gl_PrimitiveIDIn/10 
   //  uint photon_id = flags.w << 16 | flags.z ;


    uint valid  = (uint(p0.w > 0.)  << 0) + (uint(p1.w > 0.) << 1) + (uint(p1.w > p0.w) << 2) ; 
    uint select = (uint(tc > p0.w ) << 0) + (uint(tc < p1.w) << 1) + (uint(Pick.w == 0 || gl_PrimitiveIDIn/10 == Pick.w) << 2) ;  
    //(1 << 2) ;
    //
    //
    // TODO: 
    // * check ID correspondence to record array , try draw arrays approach to picking : maybe much more efficient 
    // * rejig flags and provide gui for selection based on them
    //

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

} 
