cxr_min_gl_FragDepth_issue_potentially
===========================================


Issue
-------

In some situations OpenGL event record rendering of photons seems to disappear when
infront of ray traced geometry with cxr_min.sh. This does not seem to happen with  ssst.sh
OpenGL wireframe rendering of triangulated geometry in which case eveything is done by OpenGL.
Wide views over long distances seem more susceptible making me think it could be 
a problem of zdepth precision.


* Is gl_FragDepth being used ?

  * only for raytraced geometry image

* Is FragDepth being set to correct (and consistently calculated) zdepth for both rasterized and raytraced renders ?

  * currently are assuming my OptiX calculated zdepth matches the default OpenGL calc

* Is there a zdepth precision issue ?

  * seems quite likely explanation





OpenGL default fragment depth
------------------------------

* https://paroj.github.io/gltut/Illumination/Tut13%20Deceit%20in%20Depth.html



pixel.w is only uchar 8 bits of precision : 255 possible values : that might explain issue
-----------------------------------------------------------------------------------------------

* https://www.sjbaker.org/steve/omniv/love_your_z_buffer.html


* https://stackoverflow.com/questions/71607590/increased-precision-when-rendering-depth-map


* https://nlguillemot.wordpress.com/2016/12/07/reversed-z-in-opengl/


DONE : see if moving near and far changes the extent of the issue
------------------------------------------------------------------

* increasing near seemed to do little, but pulling far in to an extreme degree 
  did make z-ordering look more correct just before the geometry and photon got 
  far clipped



TODO : try OptiX rendering to float4 rather than uchar4
----------------------------------------------------------



gl_FragDepth
-------------

::

    (ok) A[blyth@localhost ~]$ opticks-f gl_FragDepth
    ./examples/UseShaderSGLFW/UseShaderSGLFW.cc:gl_FragDepth = 0.1 ;
    ./examples/UseShaderSGLFW/UseShaderSGLFW.cc:gl_FragDepth = 0.5 ;
    ./examples/UseShaderSGLFW/UseShaderSGLFW.cc:    NB omitting a suitable setting for gl_FragDepth
    ./externals/optixnote.bash:        gl_FragDepth  = 0.5*(-A*depth + B) / depth + 0.5;
    ./sysrap/SGLDisplay.h:    gl_FragDepth = pixel.w ; 
    (ok) A[blyth@localhost opticks]$ 



gl_FragDepth ref
------------------

* https://registry.khronos.org/OpenGL-Refpages/gl4/html/gl_FragDepth.xhtml

gl_FragDepth — establishes a depth value for the current fragment

Available only in the fragment language, gl_FragDepth is an output variable
that is used to establish the depth value for the current fragment. If depth
buffering is enabled and no shader writes to gl_FragDepth, then the fixed
function value for depth will be used (this value is contained in the z
component of gl_FragCoord) otherwise, the value written to gl_FragDepth is
used. If a shader statically assigns to gl_FragDepth, then the value of the
fragment's depth may be undefined for executions of the shader that don't take
that path. That is, if the set of linked fragment shaders statically contain a
write to gl_FragDepth, then it is responsible for always writing it. 


google
--------

* https://stackoverflow.com/questions/10264949/glsl-gl-fragcoord-z-calculation-and-setting-gl-fragdepth

::

    float far=gl_DepthRange.far; float near=gl_DepthRange.near;

    vec4 eye_space_pos = gl_ModelViewMatrix * /*something*/
    vec4 clip_space_pos = gl_ProjectionMatrix * eye_space_pos;

    float ndc_depth = clip_space_pos.z / clip_space_pos.w;

    float depth = (((far-near) * ndc_depth) + near + far) / 2.0;
    gl_FragDepth = depth;



OpenGL when do you need to set gl_FragDepth - Google Search
-------------------------------------------------------------

::

    Mixed Rendering Techniques: If you are combining different rendering methods
    where the depth of certain objects needs to be precisely controlled to interact
    correctly with other elements (e.g., ray-traced spheres with other geometry).


I am mixing ray-traced with rasterized, so if my calc does not match
the OpenGL default. 


Unless you’re mixing ray-traced spheres with other geometry which is using
gl_FragCoord.z as the depth value, the calculation of gl_FragDepth isn’t
particularly important so long as it is monotonic (i.e. fragments farther from
the viewpoint have greater depth values). You can use eye-space -Z (note:
negative) or the interpolation parameter (dist_minus)

If you do need gl_FragDepth to be consistent with gl_FragCoord.z, transform the
eye-space position by the projection matrix, divide Z by W, then convert from
[-1,1] to [0,1]::

    vec4 tpoint =  projection * point;
    gl_FragDepth = (tpoint.z/tpoint.w+1.0)*0.5;







ray traced pixel.w is directly passed into gl_FragDepth
----------------------------------------------------------

sysrap/SGLDisplay.h::

    083     static constexpr const char* s_frag_source_with_FragDepth = R"(
     84 #version 330 core
     85 
     86 // samples texture at UV coordinate
     87 
     88 in vec2 UV;
     89 out vec3 color;
     90 
     91 uniform sampler2D render_tex;
     92 
     93 void main()
     94 {
     95     vec4 pixel = texture( render_tex, UV ).xyzw ;
     96     color = pixel.xyz ; 
     97     gl_FragDepth = pixel.w ; 
     98 }
     99 )";


    252 inline void SGLDisplay::init()
    253 {
    254     GLuint m_vertex_array;
    255     GL_CHECK( glGenVertexArrays(1, &m_vertex_array ) );
    256     GL_CHECK( glBindVertexArray( m_vertex_array ) );
    257 
    258     //m_program = CreateGLProgram( s_vert_source, s_frag_source );
    259     m_program = CreateGLProgram( s_vert_source, s_frag_source_with_FragDepth );
    260 
    261     m_render_tex_uniform_loc = glGetUniformLocation( m_program, "render_tex" );
    262     assert( m_render_tex_uniform_loc > -1 );
    263 




pixel.w
--------

CSGOptiX7.cu::

    162 __forceinline__ __device__ uchar4 make_normal_pixel( const float3& normal, float depth )  // pure
    163 {
    164     return make_uchar4(
    165             static_cast<uint8_t>( clamp( normal.x, 0.0f, 1.0f ) *255.0f ),
    166             static_cast<uint8_t>( clamp( normal.y, 0.0f, 1.0f ) *255.0f ),
    167             static_cast<uint8_t>( clamp( normal.z, 0.0f, 1.0f ) *255.0f ),
    168             static_cast<uint8_t>( clamp( depth   , 0.0f, 1.0f ) *255.0f )
    169             );
    170 }

    236     float3 diddled_normal = normalize(*normal)*0.5f + 0.5f ; // diddling lightens the render, with mid-grey "pedestal"
    237 
    238     float eye_z = -prd->distance()*dot(params.WNORM, direction) ;
    239     const float& A = params.ZPROJ.z ;
    240     const float& B = params.ZPROJ.w ;
    241     float zdepth = cameratype == 0u ? -(A + B/eye_z) : A*eye_z + B  ;  // cf SGLM::zdepth1
    242 
    243     if( prd->is_boundary_miss() ) zdepth = 0.999f ;
    244     // setting miss zdepth to 1.f give black miss pixels, 0.999f gives expected mid-grey from normal of (0.f,0.f,0.f)
    245     // previously with zdepth of zero for miss pixels found that OpenGL record rendering did not
    246     // appear infront of the grey miss pixels : because they were behind them (zdepth > 0.f ) presumably
    247 
    248     unsigned index = idx.y * params.width + idx.x ;
    249 
    250     if(params.pixels)
    251     {
    252 #if defined(DEBUG_PIDX)
    253         //if(idx.x == 10 && idx.y == 10) printf("//CSGOptiX7.cu:render/params.pixels diddled_normal(%7.3f,%7.3f,%7.3f)  \n", diddled_normal.x, diddled_normal.y, diddled_normal.z );
    254 #endif
    255         params.pixels[index] = params.rendertype == 0 ? make_normal_pixel( diddled_normal, zdepth ) : make_zdepth_pixel( zdepth ) ;
    256     }






