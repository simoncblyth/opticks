
// geometry coloring incld into nrm/vert.glsl and inrm/vert.glsl
//
// switch between these with "E" key  Composition::nextGeometryStyle
//
//    DEF_GEOMETRY     : simple lighting shader
//    NRMCOL_GEOMETRY  : normal shader
//    VTXCOL_GEOMETRY  : flat vertex coloring 
//    FACECOL_GEOMETRY : face coloring using psychedelic indexed colors from texture
//     
//
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
        case 3:
                colour = vec4( 1.0, 1.0, 1.0, 1.0 );  // placeholder, gets set in frag shader
                break ; 
        default:
                // NOT EXPECTED TO COME HERE
                colour = vec4( 1.0, 0.0, 0.0, 1.0 - Param.z );
                break; 
    }    



    //  gl_PrimitiveID : available in fragment shader
    //  gl_PrimitiveIDIn : 
    //
    //
    //  This approach leads to colors too close to distinguish
    //  http://stackoverflow.com/questions/28910861/opengl-unique-color-for-each-triangle
    // float r, g, b;
    // r = (gl_PrimitiveID % 255) / 255.0f;
    // g = ((gl_PrimitiveID / 255) % 255) / 255.0f;
    // b = ((gl_PrimitiveID / (255 * 255)) % 255) / 255.0f;
    //
    // frag_colour = vec4(r, g, b, 1.0);
    //
    //
    //          
    // frag_colour = texture(Colors, float(64 + gl_PrimitiveID % int(ColorDomain.z))/ColorDomain.y ) ;
    //


