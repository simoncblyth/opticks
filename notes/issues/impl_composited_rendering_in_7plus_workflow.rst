impl_composited_rendering_in_7plus_workflow
==============================================


Related
---------

* :doc:`equirectangular_camera_blackholes_sensitive_to_far`


Review old oglrap impl : "CompositeRender"
--------------------------------------------

::

    P[blyth@localhost opticks]$ opticks-f CompositeRender
    ./oglrap/Interactor.cc:    bool composite = m_composition->isCompositeRender() ; 
    ./oglrap/OpticksViz.cc:    if(m_composition->isRaytracedRender() || m_composition->isCompositeRender()) 
    ./oglrap/Scene.cc:    bool composite = m_composition->isCompositeRender() ;
    ./oglrap/Scene.cc:    bool composite = m_composition->isCompositeRender() ;
    ./optickscore/Composition.cc:bool Composition::isCompositeRender() const {   return m_render_style->isCompositeRender() ; }
    ./optickscore/Composition.hh:        bool isCompositeRender() const ;
    ./optickscore/RenderStyle.cc:bool RenderStyle::isCompositeRender() const 
    ./optickscore/RenderStyle.hh:        bool isCompositeRender() const ;
    ./opticksgl/OKGLTracer.cc:    430     if(m_scene->isRaytracedRender() || m_scene->isCompositeRender())
    P[blyth@localhost opticks]$ 


::

    613 void Interactor::nextRenderStyle(unsigned modifiers)
    614 {
    615     m_composition->nextRenderStyle(modifiers);
    616     bool composite = m_composition->isCompositeRender() ;
    617     m_scene->setSkipGeoStyle( composite ? 1 : 0) ;
    618     // inhibit rasterized geometry in raytrace composite mode 
    619 }


    0988 void Scene::renderGeometry()
     989 {
     990     if(m_skipgeo_style == NOSKIPGEO )
     991     {
     992         if(*m_global_mode_ptr && m_global_renderer)       m_global_renderer->render();
     993         if(*m_globalvec_mode_ptr && m_globalvec_renderer) m_globalvec_renderer->render();
     994         // hmm this could be doing both global and globalvec ? Or does it need to be so ?
     995 
     996 
     997         for(unsigned int i=0; i<m_num_instance_renderer; i++)
     998         {
     999             if(m_instance_mode[i] && m_instance_renderer[i]) m_instance_renderer[i]->render();
    1000             if(m_bbox_mode[i] && m_bbox_renderer[i])         m_bbox_renderer[i]->render();
    1001         }
    1002     }
    1003 
    1004     if(m_axis_mode && m_axis_renderer)     m_axis_renderer->render();
    1005 }
    1006 
    1007 
    1008 void Scene::renderEvent()
    1009 {
    1010     if(m_skipevt_style == NOSKIPEVT )
    1011     {
    1012         if(m_genstep_mode && m_genstep_renderer)  m_genstep_renderer->render();
    1013         if(m_nopstep_mode && m_nopstep_renderer)  m_nopstep_renderer->render();
    1014         if(m_photon_mode  && m_photon_renderer)   m_photon_renderer->render();
    1015         if(m_source_mode  && m_source_renderer)   m_source_renderer->render();
    1016         if(m_record_mode)
    1017         {
    1018             Rdr* rdr = getRecordRenderer();
    1019             if(rdr)
    1020                 rdr->render();
    1021         }
    1022     }
    1023 }


::

     590     //LOG(info) << "Scene::init geometry_renderer ctor DONE";
     591 
     592     m_axis_renderer = new Rdr(m_device, "axis", m_shader_dir, m_shader_incl_path );
     593 
     594     m_genstep_renderer = new Rdr(m_device, "p2l", m_shader_dir, m_shader_incl_path);
     595 
     596     m_nopstep_renderer = new Rdr(m_device, "nop", m_shader_dir, m_shader_incl_path);
     597     m_nopstep_renderer->setPrimitive(Rdr::LINE_STRIP);
     598 
     599     m_photon_renderer = new Rdr(m_device, "pos", m_shader_dir, m_shader_incl_path );
     600 
     601     m_source_renderer = new Rdr(m_device, "pos", m_shader_dir, m_shader_incl_path );
     602 
     603 
     604     //
     605     // RECORD RENDERING USES AN UNPARTIONED BUFFER OF ALL RECORDS
     606     // SO THE GEOMETRY SHADERS HAVE TO THROW INVALID STEPS AS DETERMINED BY
     607     // COMPARING THE TIMES OF THE STEP PAIRS  
     608     // THIS MEANS SINGLE VALID STEPS WOULD BE IGNORED..
     609     // THUS MUST SUPPLY LINE_STRIP SO GEOMETRY SHADER CAN GET TO SEE EACH VALID
     610     // VERTEX IN A PAIR
     611     //
     612     // OTHERWISE WILL MISS STEPS
     613     //
     614     //  see explanations in gl/altrec/geom.glsl
     615     //
     616     m_record_renderer = new Rdr(m_device, "rec", m_shader_dir, m_shader_incl_path );
     617     m_record_renderer->setPrimitive(Rdr::LINE_STRIP);
     618 
     619     m_altrecord_renderer = new Rdr(m_device, "altrec", m_shader_dir, m_shader_incl_path);
     620     m_altrecord_renderer->setPrimitive(Rdr::LINE_STRIP);
     621 
     622     m_devrecord_renderer = new Rdr(m_device, "devrec", m_shader_dir, m_shader_incl_path);
     623     m_devrecord_renderer->setPrimitive(Rdr::LINE_STRIP);
     624 
     625     m_initialized = true ;


ray trace pixels with depth 
-----------------------------

optixrap/cu/material1_radiance.cu::

     33 rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
     34 rtDeclareVariable(float, t,            rtIntersectionDistance, );
     38 rtDeclareVariable(float3,        front, , );     // normalized look direction, fed in by OTracer::trace_
     ..
     53 RT_PROGRAM void closest_hit_radiance()
     54 {
     55     const float3 n = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometricNormal)) ;
     56     const float cos_theta = dot(n,ray.direction);
     57 
     58     float intensity = 0.5f*(1.0f-cos_theta) ;  // lambertian 
     59 
     60     float zHit_eye = -t*dot(front, ray.direction) ;   // intersect z coordinate (eye frame), always -ve 

     // front 
     //     normalized world frame camera direction 


     61     float zHit_ndc = cameratype == 0u ? -ZProj.z - ZProj.w/zHit_eye : ZProj.z*zHit_eye + ZProj.w ;  // should be in range -1:1 for visibles
     62     float zHit_clip = 0.5f*zHit_ndc + 0.5f ;   // 0:1 for visibles
     63 
     64     //rtPrintf("closest_hit_radiance t %10.4f zHit_eye %10.4f  ZProj.z %10.4f ZProj.w %10.4f zHit_ndc %10.4f zHit_clip %10.4f \n", t, zHit_eye, ZProj.z, ZProj.w , zHit_ndc, zHit_clip );
     65 
     66     prd.result = make_float4(intensity, intensity, intensity, cameratype == 2u ? 0.1f : zHit_clip );
     67     // hijack .w for the depth, see notes/issues/equirectangular_camera_blackholes_sensitive_to_far.rst  
     68 
     69 #ifdef BOOLEAN_DEBUG
     70      switch(instanceIdentity.x)
     71      {
     72         case 1: prd.result.x = 1.f ; break ;
     73         case 2: prd.result.y = 1.f ; break ;
     74         case 3: prd.result.z = 1.f ; break ;
     75     }
     76 #endif
     77 
     78     prd.flag   = instanceIdentity.y ;   //  hijacked to become the hemi-pmt intersection code
     79 }


* depth info zHit_clip written to prd.result.w 


ACCORDING TO ABOVE FOR camertype == 0u (perspective?)::

    zHit_ndc = -ZProj.z  - ZProj.w/zHit_eye   

             =     ZProj.z zHit_eye + ZProj.w 
                  -------------------------------
                           -ZHit_eye

    Compare with http://www.songho.ca/opengl/gl_projectionmatrix.html

    A = ZProj.z   =?   -(f+n)/(f-n)
    B = ZProj.w   =?    -2fn/(f-n)




front + ZProj : crucial for calculating pixel depth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

front
   world frame normalized camera direction 


::

    113 double OTracer::trace_()
    114 {
    115     LOG(debug) << "OTracer::trace_ " << m_trace_count ;
    116 
    117     double t0 = BTimeStamp::RealTime();  // THERE IS A HIGHER LEVEL WAY TO DO THIS
    118 
    119     glm::vec3 eye ;
    120     glm::vec3 U ;
    121     glm::vec3 V ;
    122     glm::vec3 W ;
    123     glm::vec4 ZProj ;
    124 
    125     m_composition->getEyeUVW(eye, U, V, W, ZProj); // must setModelToWorld in composition first
    126 
    127     unsigned cameratype = m_composition->getCameraType();  // 0:PERSP, 1:ORTHO, 2:EQUIRECT
    128     unsigned pixeltime_style = m_composition->getPixelTimeStyle() ;
    129     float    pixeltime_scale = m_composition->getPixelTimeScale() ;
    130     float      scene_epsilon = m_composition->getNear();
    131 
    132     const glm::vec3 front = glm::normalize(W);
    133 
    134     m_context[ "cameratype"]->setUint( cameratype );
    135     m_context[ "pixeltime_style"]->setUint( pixeltime_style );
    136     m_context[ "pixeltime_scale"]->setFloat( pixeltime_scale );
    137     m_context[ "scene_epsilon"]->setFloat(scene_epsilon);
    138     m_context[ "eye"]->setFloat( make_float3( eye.x, eye.y, eye.z ) );
    139     m_context[ "U"  ]->setFloat( make_float3( U.x, U.y, U.z ) );
    140     m_context[ "V"  ]->setFloat( make_float3( V.x, V.y, V.z ) );
    141     m_context[ "W"  ]->setFloat( make_float3( W.x, W.y, W.z ) );
    142     m_context[ "front"  ]->setFloat( make_float3( front.x, front.y, front.z ) );
    143     m_context[ "ZProj"  ]->setFloat( make_float4( ZProj.x, ZProj.y, ZProj.z, ZProj.w ) );
    144 






    2314 /**
    2315 Composition::getEyeUVW
    2316 ------------------------
    2317    
    2318 Eye frame axes and origin transformed into world frame
    2319 
    2320 
    2321           top        
    2322                    gaze
    2323             +Y    -Z 
    2324              |    /
    2325              |   /
    2326              |  /
    2327              | /
    2328              |/
    2329              O--------- +X   right
    2330             /
    2331            /
    2332           /
    2333          /
    2334        +Z
    2335 
    2336 **/
    2337 
    2338 void Composition::getEyeUVW(glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W, glm::vec4& ZProj )
    2339 {
    2340     update();
    2341 
    2342     float length = getLength();
    2343     float tanYfov = m_camera->getTanYfov();  // reciprocal of camera zoom
    2344     float aspect = m_camera->getAspect();
    2345 
    2346     m_camera->fillZProjection(ZProj); // 3rd row of projection matrix
    2347 
    2348     float v_half_height = length * tanYfov ;
    2349     float u_half_width  = v_half_height * aspect ;
    2350     float w_depth       = m_gazelength ;
    2351 
    2352     glm::vec4 right( 1., 0., 0., 0.);
    2353     glm::vec4   top( 0., 1., 0., 0.);
    2354     glm::vec4  gaze( 0., 0.,-1., 0.);   // towards -Z
    2355     glm::vec4 origin(0., 0., 0., 1.);
    2356 
    2357     // and scaled to focal plane dimensions 
    2358 
    2359     U = glm::vec3( m_eye2world * right ) * u_half_width ;
    2360     V = glm::vec3( m_eye2world * top   ) * v_half_height ;
    2361     W = glm::vec3( m_eye2world * gaze  ) * w_depth  ;
    2362     eye = glm::vec3( m_eye2world * origin );
    2363 }

    641 glm::mat4 Camera::getProjection() const
    642 {
    643     return isOrthographic() ? getOrtho() : getFrustum() ;
    644 }
    645 
    646 
    647 void Camera::fillZProjection(glm::vec4& zProj) const
    648 {
    649     glm::mat4 proj = getProjection() ;
    650     zProj.x = proj[0][2] ;
    651     zProj.y = proj[1][2] ;
    652     zProj.z = proj[2][2] ;
    653     zProj.w = proj[3][2] ;
    654 }
    655 
    656 glm::mat4 Camera::getPerspective() const
    657 {
    658     return glm::perspective(getYfov(), getAspect(), getNear(), getFar());
    659 }
    660 glm::mat4 Camera::getOrtho() const
    661 {
    662     return glm::ortho( getLeft(), getRight(), getBottom(), getTop(), getNear(), getFar() );
    663 }
    664 glm::mat4 Camera::getFrustum() const
    665 {
    666     return glm::frustum( getLeft(), getRight(), getBottom(), getTop(), getNear(), getFar() );
    667 }


SGLM.h equivalent
-------------------

::

    1401 void SGLM::updateProjection()
    1402 {
    1403     //float fsc = get_focal_basis() ;
    1404     float fsc = get_transverse_scale() ;
    1405     float fscz = fsc/ZOOM  ;
    1406 
    1407     float aspect = Aspect();
    1408     float left   = -aspect*fscz ;
    1409     float right  =  aspect*fscz ;
    1410     float bottom = -fscz ;
    1411     float top    =  fscz ;
    1412 
    1413     float near_abs   = get_near_abs() ;
    1414     float far_abs    = get_far_abs()  ;
    1415 
    1416     assert( cam == CAM_PERSPECTIVE || cam == CAM_ORTHOGRAPHIC );
    1417     switch(cam)
    1418     {
    1419        case CAM_PERSPECTIVE:  projection = glm::frustum( left, right, bottom, top, near_abs, far_abs ); break ;
    1420        case CAM_ORTHOGRAPHIC: projection = glm::ortho( left, right, bottom, top, near_abs, far_abs )  ; break ;
    1421     }
    1422 }


OpenGL DEPTH
----------------

::

    P[blyth@localhost oglrap]$ grep DEPTH *.*
    Frame.cc:    glEnable(GL_DEPTH_TEST);
    Frame.cc:     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Frame.cc:     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Frame.cc:Access the GL_DEPTH_COMPONENT float for a pixel position in the frame.
    Frame.cc:    GLenum format = GL_DEPTH_COMPONENT ; 
    GUI.cc:    glEnable(GL_DEPTH_TEST);
    Scene.cc:Called with pixel coordinates and z-depth float from GL_DEPTH_COMPONENT (0:1)
    Texture.cc:         format = GL_DEPTH_COMPONENT ;
    P[blyth@localhost oglrap]$ 


    0344 void Frame::initContext()
     345 {
     346     // hookup the callbacks and arranges outcomes into event queue 
     347     gleqTrackWindow(m_window);
     348 
     349 
     350 
     351     // start GLEW extension handler, segfaults if done before glfwCreateWindow
     352     glewExperimental = GL_TRUE;
     353     glewInit ();
     354 
     355     GLenum err = glGetError();   // getting the error should clear it 
     356     assert( err == GL_INVALID_ENUM ) ; // long standing glew bug, see https://learnopengl.com/In-Practice/Debugging
     357     err = glGetError();
     358     assert( err == GL_NO_ERROR );
     359 
     360 
     361     G::ErrCheck("Frame::initContext.[", true);
     362     G::ErrCheck("Frame::initContext.1", true);
     363 
     364 
     365     glEnable(GL_DEPTH_TEST);
     366     G::ErrCheck("Frame::initContext.2", true);
     367 
     368     glDepthFunc(GL_LESS);  // overwrite if distance to camera is less
     369     G::ErrCheck("Frame::initContext.3", true);
     370 

     492 void Frame::clear()
     493 {
     494      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
     495 }

     533 /**
     534 Frame::readDepth
     535 -----------------
     536 
     537 Access the GL_DEPTH_COMPONENT float for a pixel position in the frame.
     538 The value is expected to be in the range 0:1
     539 
     540 **/
     541 
     542 float Frame::readDepth( int x, int y_, int yheight )
     543 {
     544     GLint y = yheight - y_ ;
     545     GLsizei width(1), height(1) ;
     546     GLenum format = GL_DEPTH_COMPONENT ;
     547     GLenum type = GL_FLOAT ;
     548     float depth ;
     549     glReadPixels(x, y, width, height, format, type, &depth );
     550     return depth ;
     551 }

    719 void GUI::render()
    720 {
    721     ImGuiIO& io = ImGui::GetIO();
    722     glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);
    723     ImGui::Render();
    724 
    725     // https://github.com/ocornut/imgui/issues/109
    726     // fix ImGui diddling of OpenGL state
    727     glDisable(GL_BLEND);
    728     //glEnable(GL_CULL_FACE);  going-one-sided causes issues
    729     glEnable(GL_DEPTH_TEST);
    730 }


    1085 /**
    1086 Scene::touch
    1087 --------------
    1088 
    1089 Called with pixel coordinates and z-depth float from GL_DEPTH_COMPONENT (0:1)
    1090 returns the index of the smallest volume that contains the point. 
    1091 
    1092 Formerly this made the problematic "all volume" assumption for mm0, 
    1093 now fixed by migrating to GNodeLib. 
    1094 
    1095 **/
    1096 
    1097 int Scene::touch(int ix, int iy, float depth)
    1098 {
    1099     assert( m_nodelib && "m_nodelib must not be NULL");
    1100 
    1101     glm::vec3 tap = m_composition->unProject(ix,iy, depth);
    1102 
    1103     //gfloat3 gt(t.x, t.y, t.z );
    1104     //int container = m_mesh0->findContainer(gt);
    1105 
    1106     int container = m_nodelib->findContainerVolumeIndex(tap.x, tap.y, tap.z);
    1107 
    1108     LOG(LEVEL)
    1109         << " x " << tap.x
    1110         << " y " << tap.y
    1111         << " z " << tap.z
    1112         << " containerVolumeIndex " << container
    1113         ;
    1114 
    1115    if(container > 0) setTouch(container);
    1116    return container ;
    1117 }


Depth investigations enabled with envvar SGLFW__DEPTH
------------------------------------------------------

Added saving of GL_DEPTH_COMPONENT pixels to SGLFW.h by generalizing SIMG_Frame.h

* single channel _depth.jpg from SGLFW_SOPTIX_Scene_test.sh are very high key
  can only vaguely see the depth info. Using imshow on the _depth.npy is clearer. 

* doing the same with the ray traced CSGOptiXRenderInteractiveTest gives
  a uniform grey depth map : everything at same depth (probably all zero ? why not black?) 

* potentially issue with greyscale jpg compression ? So until learn more look at the _depth.npy

::

    P[blyth@localhost issues]$ o
    On branch master
    Your branch is up to date with 'origin/master'.

    Changes to be committed:
      (use "git restore --staged <file>..." to unstage)
        deleted:    sysrap/tests/ssst1.sh

    Changes not staged for commit:
      (use "git add <file>..." to update what will be committed)
      (use "git restore <file>..." to discard changes in working directory)
        modified:   CSGOptiX/CSGOptiX.cc
        modified:   CSGOptiX/Frame.h
        modified:   CSGOptiX/cxr_min.sh
        modified:   sysrap/SGLFW.h
        modified:   sysrap/SIMG_Frame.h
        modified:   sysrap/tests/SGLFW_SOPTIX_Scene_test.sh
        modified:   sysrap/tests/SIMGTest.py

    Untracked files:
      (use "git add <file>..." to include in what will be committed)
        sysrap/tests/SGLFW__DEPTH.py
        sysrap/tests/SGLFW__DEPTH.sh

    P[blyth@localhost opticks]$ 


OpenGL add GL_DEPTH_COMPONENT to ray traced pixels ?
-----------------------------------------------------

* https://stackoverflow.com/questions/13804794/render-2d-image-with-depth-in-opengl-preserving-depth-testing

Load your depth map via glDrawPixels(..., ..., GL_DEPTH_COMPONENT, ..., ...) and render as usual.

* http://www.opengl.org/sdk/docs/man2/xhtml/glDrawPixels.xml

Using OpenGL pixel_buffer_object, you can bind depth textures. So the process would be as follows:

* Load external texture
* Load external depth texture
* Create pixel_buffer_object with the two textures
* Set PBO as render target and render the rest of your geometry (don't glClear before rendering).

gl_FragDepth
---------------

https://registry.khronos.org/OpenGL-Refpages/gl4/html/gl_FragDepth.xhtml

Declaration
out float gl_FragDepth ;

Description

Available only in the fragment language, gl_FragDepth is an output variable
that is used to establish the depth value for the current fragment. If depth
buffering is enabled and no shader writes to gl_FragDepth, then the fixed
function value for depth will be used (this value is contained in the z
component of gl_FragCoord) otherwise, the value written to gl_FragDepth is
used. If a shader statically assigns to gl_FragDepth, then the value of the
fragment's depth may be undefined for executions of the shader that don't take
that path. That is, if the set of linked fragment shaders statically contain a
write to gl_FragDepth, then it is responsible for always writing it.

* https://registry.khronos.org/OpenGL-Refpages/gl4/html/gl_FragCoord.xhtml


oglrap/gl/tex/frag.glsl 
--------------------------

This is the shader formerly used to draw the ray traced pixels. 

This is crucially how I did it before, storing the OptiX/CUDA 
ray trace calculated depth in prd.result.w which then gets passed 
as texture to OpenGL for viz together with other draws that 
have natural depth. 

::

     21 #pragma debug(on)
     22 
     23 //in vec3 colour;
     24 in vec2 texcoord;
     25 
     26 out vec4 frag_colour;
     27 
     28 uniform  vec4 ScanParam ;
     29 uniform vec4 ClipPlane ;
     30 uniform ivec4 NrmParam ;
     31 
     32 uniform sampler2D ColorTex ;
     33 
     34 void main ()
     35 {
     36    frag_colour = texture(ColorTex, texcoord);
     37    float depth = frag_colour.w ;  // alpha is hijacked for depth in pinhole_camera.cu material1_radiance.cu
     38    frag_colour.w = 1.0 ;
     39 
     40    gl_FragDepth = depth  ;
     41 
     42    if(NrmParam.z == 1)
     43    {
     44         if(depth < ScanParam.x || depth > ScanParam.y ) discard ;
     45    }
     46 }
     47 
     48 
     49 //
     50 //  the input color is ignored
     51 //
     52 //
     53 // http://www.roxlu.com/2014/036/rendering-the-depth-buffer
     54 //
     55 // gl_FragDepth = 1.1 ;   // black
     56 // gl_FragDepth = 1.0 ;   // black
     57 // gl_FragDepth = 0.999 ; //  visible geometry
     58 // gl_FragDepth = 0.0   ; //  visible geometry
     59 //
     60 // frag_colour = vec4( depth, depth, depth, 1.0 );
     61 // vizualize fragment depth, the closer you get to geometry the darker it gets
     62 // reaching black just before being near clipped
     63 //



ray tracing calculate z-depth : from Inigo Quilez (of SDF fame)
-------------------------------------------------------------------

* https://iquilezles.org/articles/raypolys/

This is basic stuff, but why not to refresh memory from time to time and
revisit the basic concepts once again. Say you are raymarching or raytracing
some objects in a fragment shader, and you want to composite them with some
other geometry that you rendered or will render through regular rasterization.
The only thing you need to do is to output a depth value in your
raytracing/marching shader, and let the depth buffer do the rest. The first to
do, then, is to understand what "depth" means here.

In a raytracer/marcher, you probably have access to the distance from the ray
origin (you camera position) to the closest geometry/intersection point. That
distance is not what you want to write to the depth buffer, as hardware
rasterizers (OpenGL or DirectX) don't store distances to the camera, but the z
of the geometry/intersection point. The reason is that this z value is still
monotonically increasing with the distance, but has the property of being
linear (linear like in "can be interpolated across the surface of a play 3D
triangle). So, in your raymarcher, compute the intersection point, and use its
z component for writing to the depth buffer.

::

   eye frame origin is the "camera" position looking along along -Z axis  
   so the relevant z is eye frame z (which will be -ve)

Well, that will not work just like that. your api of preference will remap your
z values to a -1 to 1 range based on the near and far clipping planes you
decided to set up. Furthermore, the remapping will probably also transform your
z values to some other sort of scale that exploits the properties of
perspective (like with a curve that compresses values in the far distance). So
you will have to implement the same remapping in your shader before you can
merge your raytraced/marched objects with the rest of the polygons.

The mapping is simple, though, and is normally configured by the projection
matrix. Grab your OpenGL Redbook, and have a look to the content of a standard
projection matrix. The third and fourth row are what we need, since those are
the ones that affect the z and w components of your points when transformed
from eye to clip space. So, if ze is the z of your intersection point in camera
(eye) space, then you can compute the clip space z and w as

zc = -ze*(far+near)/(far-near) - 2*far*near/(far-near)
wc = -ze




The hardware will then do the perspective division and compute the z value in
normalized device coordinates before converting it to a 24 bit depth value:

zn = zc/zw = (far+near)/(far-near) + 2*far*near/(far-near)/ze


SCB correction + extension (comparing with http://www.songho.ca/opengl/gl_projectionmatrix.html)::


    zn = zc/wc = (far+near)/(far-near) + 2*far*near/(far-near)/ze
        


     zn = zc/wc =  (far+near) + 2*far*near/ze     
                   ------------------------------  
                          (far-near)


      zn [ ze = -near ] =  (far+near) - 2*far   =   near - far    = -1 
                           ------------------      ------------
                                 far - near         far - near 

      zn [ ze = -far ] =  (far+near) - 2*near   =   far - near   = +1 
                           ------------------      ------------
                                 far - near         far - near 



       zn * (-ze)  =  -(far+near)/(far-near) ze - 2*far*near/(far-near)



which you can see it is a formula of the form zn = a + b/ze which produces the
desired depth compression. You can check that the boundary conditions are met,
by doing

ze = -near -> zn = -1;
ze = -far -> zn = 1;

Yeah, remember that your depths in camera space are negative inwards the
screen. So, our raytracing/marching shader should end with something like

float a = (far+near)/(far-near);
float b = 2.0*far*near/(far-near);
gl_FragDepth = a + b/z;

You probably want to upload a and b as uniforms to your shader.

Alternatively, if you don't want mess with all this, you can directly grab the
projection parameters from the projection matrix, and do something like

float zc = ( (ModelView)ProjectionMatrix * vec4( intersectionPoint, 1.0 ) ).z;
float wc = ( (ModelView)ProjectionMatrix * vec4( intersectionPoint, 1.0 ) ).w;
gl_FragDepth = zc/wc;

which is a little bit more expensive, but gives the same results...





Ingredients
--------------

* OpenGL shaders that are passed event data such as records and produce rasterized representations

* attributes that describe the event data in form that OpenGL can understand such that 
  can access positions and times from the OpenGL shaders 

* ray traced pixels with depth information, calculated using the raster projection corresponding 
  to the ray trace params, even though ray tracing does not use a projection matrix

* OpenGL compositing setup to use the depth information in the combination of pixels, so the 
  frontmost pixels are visible 



