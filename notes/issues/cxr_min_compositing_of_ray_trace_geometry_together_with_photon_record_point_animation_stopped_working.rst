cxr_min_compositing_of_ray_trace_geometry_together_with_photon_record_point_animation_stopped_working
=======================================================================================================


Issue
-----

* compositing geom render from optix ray trace and event render from OpenGL geometry shader not working like it did with old-opticks ?

Reproduce
----------

For simple geometry and photon record testing of compositing see::

    g4cx/tests/G4CXTest_raindrop_animation.sh

SGLFW has X mode, which toggles rendertype between normal and zdepth display.
That was probably implemented to assist with debugging this longstanding issue


POSSIBLE FIX : compositing improved after avoid conflation of zdepth_ndc -1:1 and zdepth_clip 0:1 in CSGOptiX7.cu
------------------------------------------------------------------------------------------------------------------

* improvement is particulary obvious in orthographic projection
* perspective projection as always is more difficult to judge - but definitely improved also

::

     #if defined(DEBUG_PIDX)
    @@ -322,9 +322,11 @@ static __forceinline__ __device__ void render( const uint3& idx, const uint3& di
         float eye_z = -prd->distance()*dot(to_float3(params.WNORM), direction) ;
         const float& A = params.ZPROJ.z ;
         const float& B = params.ZPROJ.w ;
    -    float zdepth = cameratype == 0u ? -(A + B/eye_z) : A*eye_z + B  ;  // cf SGLM::zdepth1
     
    -    if( prd->is_boundary_miss() ) zdepth = 0.999f ;
    +    float zdepth_ndc  = cameratype == 0u ? -(A + B/eye_z) : A*eye_z + B  ;  // cf SGLM::zdepth1
    +    float zdepth_clip = 0.5f*(zdepth_ndc + 1.f);
    +
    +    if( prd->is_boundary_miss() ) zdepth_clip = 0.999f ;
         // setting miss zdepth to 1.f give black miss pixels, 0.999f gives expected mid-grey from normal of (0.f,0.f,0.f)
         // previously with zdepth of zero for miss pixels found that OpenGL record rendering did not
         // appear infront of the grey miss pixels : because they were behind them (zdepth > 0.f ) presumably
    @@ -336,7 +338,8 @@ static __forceinline__ __device__ void render( const uint3& idx, const uint3& di
     #if defined(DEBUG_PIDX)
             //if(idx.x == 10 && idx.y == 10) printf("//CSGOptiX7.cu:render/params.pixels diddled_normal(%7.3f,%7.3f,%7.3f)  \n", diddled_normal.x, diddled_normal.y, diddled_normal.z );
     #endif
    -        params.pixels[index] = params.rendertype == 0 ? make_normal_pixel( diddled_normal, zdepth ) : make_zdepth_pixel( zdepth ) ;
    +        //params.pixels[index] = params.rendertype == 0 ? make_normal_pixel( diddled_normal, zdepth_clip ) : make_zdepth_pixel( zdepth_clip, params.DEBUG_zdepth == 0.f ? zdepth_clip : params.DEBUG_zdepth ) ;
    +        params.pixels[index] = params.rendertype == 0 ? make_normal_pixel( diddled_normal, zdepth_clip ) : make_zdepth_pixel( zdepth_clip, zdepth_clip ) ;
         }








presumably inconsistent zdepth calculation between OpenGL and the ray tracer ?
--------------------------------------------------------------------------------

::

    313 
    314     const float3* normal = prd->normal();
    315 
    316 #if defined(DEBUG_PIDX)
    317     //if(idx.x == 10 && idx.y == 10) printf("//CSGOptiX7.cu:render normal(%7.3f,%7.3f,%7.3f)  \n", normal->x, normal->y, normal->z );
    318 #endif
    319 
    320     float3 diddled_normal = normalize(*normal)*0.5f + 0.5f ; // diddling lightens the render, with mid-grey "pedestal"
    321 
    322     float eye_z = -prd->distance()*dot(to_float3(params.WNORM), direction) ;
    323     const float& A = params.ZPROJ.z ;
    324     const float& B = params.ZPROJ.w ;
    325     float zdepth = cameratype == 0u ? -(A + B/eye_z) : A*eye_z + B  ;  // cf SGLM::zdepth1
    326 
    327     if( prd->is_boundary_miss() ) zdepth = 0.999f ;
    328     // setting miss zdepth to 1.f give black miss pixels, 0.999f gives expected mid-grey from normal of (0.f,0.f,0.f)
    329     // previously with zdepth of zero for miss pixels found that OpenGL record rendering did not
    330     // appear infront of the grey miss pixels : because they were behind them (zdepth > 0.f ) presumably
    331 
    332     unsigned index = idx.y * params.width + idx.x ;
    333 
    334     if(params.pixels)
    335     {
    336 #if defined(DEBUG_PIDX)
    337         //if(idx.x == 10 && idx.y == 10) printf("//CSGOptiX7.cu:render/params.pixels diddled_normal(%7.3f,%7.3f,%7.3f)  \n", diddled_normal.x, diddled_normal.y, diddled_normal.z );
    338 #endif
    339         params.pixels[index] = params.rendertype == 0 ? make_normal_pixel( diddled_normal, zdepth ) : make_zdepth_pixel( zdepth ) ;
    340     }



rendertype
-----------

::

    [lo] A[blyth@localhost sysrap]$ opticks-f rendertype
    ./CSGOptiX/CSGOptiX7.cu:        params.pixels[index] = params.rendertype == 0 ? make_normal_pixel( diddled_normal, zdepth ) : make_zdepth_pixel( zdepth ) ;
    ./CSGOptiX/Params.h:    int32_t    rendertype ;
    ./CSGOptiX/Params.h:    void setCamera(float tmin_, float tmax_, unsigned cameratype_, int traceyflip_, int rendertype_, const glm::vec4& ZPROJ_ ) ;
    ./CSGOptiX/Params.cc:    int rendertype_,
    ./CSGOptiX/Params.cc:    params->rendertype = rendertype_ ;
    ./CSGOptiX/Params.cc:       << std::setw(20) << " rendertype " << std::setw(10) << params->rendertype  << std::endl
    ./CSGOptiX/CSGOptiX.cc:    int rendertype = sglm->rendertype ;
    ./CSGOptiX/CSGOptiX.cc:    params_helper->setCamera(tmin, tmax, cameratype, traceyflip, rendertype, ZPROJ );
    ./sysrap/SGLFW.h:   [--rendertype] toggle between normal and zdepth shading
    ./sysrap/SGLFW.h:    void rendertype();
    ./sysrap/SGLFW.h:            case GLFW_KEY_X:      command("--rendertype")            ; break ;   // HMM: also in SGLM_Modnav
    ./sysrap/SGLFW.h:    if(strcmp(cmd, "--rendertype") == 0) rendertype();
    ./sysrap/SGLFW.h:inline void SGLFW::rendertype()
    ./sysrap/SGLFW.h:    gm.command("--rendertype");
    ./sysrap/SGLM.h:    void toggle_rendertype();
    ./sysrap/SGLM.h:    int   rendertype ;
    ./sysrap/SGLM.h:    rendertype(0),
    ./sysrap/SGLM.h:void SGLM::toggle_rendertype()
    ./sysrap/SGLM.h:    rendertype = !rendertype ;
    ./sysrap/SGLM.h:        else if(strcmp(op,"rendertype")==0) gm->toggle_rendertype();
    [lo] A[blyth@localhost opticks]$ 




line shader rather than flying_point would be easier to debug
--------------------------------------------------------------

::

    [lo] A[blyth@localhost opticks]$ opticks-f flying_point
    ./examples/UseGeometryShader/run.sh:    REC=1 ADHOC=0.001 T0=0 T1=200 SHADER=rec_flying_point ~/o/examples/UseGeometryShader/run.sh
    ./examples/UseGeometryShader/run.sh:    REC=0 ADHOC=0.5 T0=0 T1=10 SHADER=rec_flying_point ~/o/examples/UseGeometryShader/run.sh
    ./examples/UseGeometryShader/run.sh:    REC=0                      SHADER=rec_flying_point ~/o/examples/UseGeometryShader/run.sh
    ./examples/UseGeometryShader/UseGeometryShader.cc:    SHADER=rec_flying_point ~/o/examples/UseGeometryShader/build.sh
    ./examples/UseGeometryShader/UseGeometryShader.cc:    // The strings below are names of uniforms present in rec_flying_point/geom.glsl and pos/vert.glsl
    ./examples/UseGeometryShader/build.sh:#shader=rec_flying_point
    ./examples/UseGeometryShader/build.sh:shader=rec_flying_point_persist
    ./examples/UseGeometryShader/go.sh:export SHADER_FOLD=$sdir/rec_flying_point
    ./sysrap/SGLFW_Event_TO_BE_REMOVED.h:    rec_prog = new SGLFW_Program(spath::Resolve(shader_fold, "rec_flying_point_persist"), nullptr, nullptr, nullptr, "ModelViewProjection", gm.MVP_ptr );
    ./sysrap/tests/spath_test.cc:    const char* name_ = "${SGLFW_Evt__shader_name:-rec_flying_point_persist}" ;
    ./sysrap/SGLFW_Evt.h:    rec_shader_name = spath::Resolve("${SGLFW_Evt__rec_shader_name:-rec_flying_point_persist}") ;
    ./sysrap/populate_gl.sh:#cp -r ../examples/UseGeometryShader/rec_flying_point_persist gl/
    ./opticks.bash:shader_name=rec_flying_point
    ./opticks.bash:#shader_name=rec_flying_point_persist
    [lo] A[blyth@localhost opticks]$ 


::

    [lo] A[blyth@localhost sysrap]$ l $OPTICKS_PREFIX/gl/
    total 40
    4 drwxr-xr-x. 17 blyth blyth 4096 Jun 25 10:29 ..
    4 drwxr-xr-x.  2 blyth blyth 4096 Jul  7  2025 gen_line_strip
    4 drwxr-xr-x.  2 blyth blyth 4096 Jul  7  2025 rec_line_strip
    4 drwxr-xr-x. 10 blyth blyth 4096 Jul  6  2025 .
    4 drwxr-xr-x.  2 blyth blyth 4096 Jul  5  2025 rec_flying_point
    4 drwxr-xr-x.  2 blyth blyth 4096 Jul  2  2025 inormal
    4 drwxr-xr-x.  2 blyth blyth 4096 Jul  2  2025 iwireframe
    4 drwxr-xr-x.  2 blyth blyth 4096 Jul  2  2025 normal
    4 drwxr-xr-x.  2 blyth blyth 4096 Jul  2  2025 rec_flying_point_persist
    4 drwxr-xr-x.  2 blyth blyth 4096 Jul  2  2025 wireframe
    [lo] A[blyth@localhost sysrap]$ 



HMM maybe have conflated ndc coordinates in -1:1 with zdepth ? YEP looks so
----------------------------------------------------------------------------

zHit_ndc (perspective)  :  -A - B/ze  = -( A + B/ze )
zHit_ndc (orthographic) :   A*ze + B



::

    136      // front 
    137      //     normalized world frame camera direction 
    138 
    139 
    140      61     float zHit_ndc = cameratype == 0u ? -ZProj.z - ZProj.w/zHit_eye : ZProj.z*zHit_eye + ZProj.w ;  // should be in range -1:1 for visibles
    141      62     float zHit_clip = 0.5f*zHit_ndc + 0.5f ;   // 0:1 for visibles
    142      63 
    143      64     //rtPrintf("closest_hit_radiance t %10.4f zHit_eye %10.4f  ZProj.z %10.4f ZProj.w %10.4f zHit_ndc %10.4f zHit_clip %10.4f \n", t, zHit_eye, ZProj.z, ZProj.w , zHit_ndc, zHit_clip );
    144      65 
    145      66     prd.result = make_float4(intensity, intensity, intensity, cameratype == 2u ? 0.1f : zHit_clip );
    146      67     // hijack .w for the depth, see notes/issues/equirectangular_camera_blackholes_sensitive_to_far.rst  
    147      68 
    148      69 #ifdef BOOLEAN_DEBUG
    149      70      switch(instanceIdentity.x)
    150      71      {
    151      72         case 1: prd.result.x = 1.f ; break ;
    152      73         case 2: prd.result.y = 1.f ; break ;
    153      74         case 3: prd.result.z = 1.f ; break ;
    154      75     }
    155      76 #endif
    156      77 
    157      78     prd.flag   = instanceIdentity.y ;   //  hijacked to become the hemi-pmt intersection code
    158      79 }
    159 
    160 
    161 * depth info zHit_clip written to prd.result.w



