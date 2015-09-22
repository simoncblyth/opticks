
// geometry coloring incld into nrm/vert.glsl and inrm/vert.glsl

    switch(NrmParam.y)
    {
        case 0:
                vec3 lpos_e = vec3( ModelView * LightPosition );   
                vec3 ldir_e = normalize(lpos_e - vpos_e);
                float diffuse = clamp( dot(normal, ldir_e), 0, 1) ;
                //float diffuse =  abs(dot(normal, ldir_e)) ;   // absolution rather than clamping makes a big difference  
                colour = vec4( ambient + vec3(diffuse) * vertex_colour , 1.0 - Param.z );
                break;
        case 1:
                colour = vec4( normal*0.5 + 0.5, 1.0 - Param.z ) ;    // 1 - alpha 
                break; 
        case 2:
                colour = vec4( vertex_colour, 1.0 - Param.z ); 
                break; 
        default:
                colour = vec4( 1.0, 0.0, 0.0, 1.0 - Param.z );
                break; 
    }    


