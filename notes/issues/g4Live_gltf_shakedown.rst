g4Live_gltf_shakedown
========================


DYB Nodes for glTF check viz
--------------------------------

Debug by editing the glTF to pick particular nodes::

    329578   "scenes": [
    329579     {
    329580       "nodes": [
    329581         3199
    329582       ]
    329583     }

::

   3199 : single pmt (with frame false looks correct, with frame true mangled)
   3155 : AD  (view starts from above the lid) (with frame false PMT all pointing in one direction, with frame true correct)
   3147 : pool with 2 ADs etc..


NEXT
-----

* raytracing g4live direct geometry (analytic)

  * examine how GScene operates, possibly all thats needed is 
    some metadata in the GMergedMesh to switch geocode for oxrap.OGeo 
     


DONE : Oldschool from geocache analytic
------------------------------------------

Required a rebuild of geocache, note beautiful but slower raytrace of the analytic CSG::

    op --gltf 3 -G 
    op --gltf 3 


TODO : REJIG GGeo/GScene/GGeoTest in triplicate pattern ?
-------------------------------------------------------------

* X4 populated GGeo has both triangulated (GMesh) and analytic (GParts)
  side by side, with the GVolume(GNode) tree being common to the two 

* separate GScene from NGLTF now seems a mistake, 
  better to incorporate the analytic GParts from GScene inside  
  a partner GGeo ?

  * nevertheless : should get this route operational anyhow, before replacing it 

* GGeoTest feeding off a basis GGeo, seems less problematic 


DONE : raytracing g4live direct geometry (triangulated)
-----------------------------------------------------------

* moving OpticksHub to picking up the last GGeo instance with GGeo::GetInstance 
  allows the OKMgr to work unchanged 


FIXED : Switch to raytrace gives OptiX validation fail, for lack of record_buffer
-------------------------------------------------------------------------------------

* fixed by giving "--tracer" argument to the 2nd Opticks, which has the 
  effect of skipping "OpEngine::initPropagation" : which although it doesnt 
  fail, seems to kill the OptiX context causing a fail on the raytracer launch :
  because it expects some buffers from event upload 

::

     670 bool Opticks::isTracer() const
     671 {
     672     return m_cfg->hasOpt("tracer") ;
     673 }

::

    epsilon:optickscore blyth$ opticks-find isTracer
    ./okop/OpEngine.cc:   else if(m_ok->isTracer())
    ./optickscore/Opticks.cc:bool Opticks::isTracer() const
    ./optickscore/Opticks.hh:       bool isTracer() const;

::

     56 void OpEngine::init()
     57 {
     58    m_ok->setOptiXVersion(OConfig::OptiXVersion());
     59    if(m_ok->isLoad())
     60    {
     61        LOG(warning) << "OpEngine::init skip initPropagation as just loading pre-cooked event " ;
     62    }
     63    else if(m_ok->isTracer())
     64    {
     65        LOG(warning) << "OpEngine::init skip initPropagation as tracer mode is active  " ;
     66    }
     67    else
     68    {
     69        LOG(warning) << "OpEngine::init initPropagation START" ;
     70        initPropagation();
     71        LOG(warning) << "OpEngine::init initPropagation DONE" ;
     72 
     73    }
     74 }






OKX4Test then press O::

    2018-06-24 21:08:50.962 INFO  [26433846] [OTracer::trace_@128] OTracer::trace  entry_index 1 trace_count 0 resolution_scale 1 size(2880,1704) ZProj.zw (-1.04159,-2079.67) front 0.5971,0.6757,-0.4322
    2018-06-24 21:08:50.963 INFO  [26433846] [OContext::close@236] OContext::close numEntryPoint 2
    2018-06-24 21:08:50.964 INFO  [26433846] [OContext::close@240] OContext::close setEntryPointCount done.
    2018-06-24 21:08:51.184 INFO  [26433846] [OContext::close@246] OContext::close m_cfg->apply() done.
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Variable not found (Details: Function "RTresult _rtContextValidate(RTcontext)" caught exception: Variable "Unresolved reference to variable record_buffer from _Z8generatev_cp7" not found in scope)
    Process 58275 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff56001b6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff56001b6e <+10>: jae    0x7fff56001b78            ; <+20>
        0x7fff56001b70 <+12>: movq   %rax, %rdi
        0x7fff56001b73 <+15>: jmp    0x7fff55ff8b00            ; cerror_nocancel
        0x7fff56001b78 <+20>: retq   
    Target 0: (OKX4Test) stopped.
    (lldb) 
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff56001b6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff561cc080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff55f5d1ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff53e61f8f libc++abi.dylib`abort_message + 245
        frame #4: 0x00007fff53e62113 libc++abi.dylib`default_terminate_handler() + 241
        frame #5: 0x00007fff55299eab libobjc.A.dylib`_objc_terminate() + 105
        frame #6: 0x00007fff53e7d7c9 libc++abi.dylib`std::__terminate(void (*)()) + 8
        frame #7: 0x00007fff53e7d26f libc++abi.dylib`__cxa_throw + 121
        frame #8: 0x00000001004b9eb6 libOptiXRap.dylib`optix::ContextObj::checkError(this=0x000000012006f690, code=RT_ERROR_VARIABLE_NOT_FOUND) const at optixpp_namespace.h:1963
        frame #9: 0x00000001004b9f17 libOptiXRap.dylib`optix::ContextObj::validate(this=0x000000012006f690) at optixpp_namespace.h:2008
        frame #10: 0x00000001004ce4a8 libOptiXRap.dylib`OContext::validate_(this=0x000000012007c960) at OContext.cc:308
        frame #11: 0x00000001004cde81 libOptiXRap.dylib`OContext::launch(this=0x000000012007c960, lmode=30, entry=1, width=2880, height=1704, times=0x0000000123fc2740) at OContext.cc:275
        frame #12: 0x00000001004e09a7 libOptiXRap.dylib`OTracer::trace_(this=0x0000000129b31e90) at OTracer.cc:142
        frame #13: 0x00000001001318d5 libOpticksGL.dylib`OKGLTracer::render(this=0x0000000123fc1f70) at OKGLTracer.cc:165
        frame #14: 0x00000001001c7de1 libOGLRap.dylib`OpticksViz::render(this=0x00007ffeefbfe220) at OpticksViz.cc:432
        frame #15: 0x00000001001c69f2 libOGLRap.dylib`OpticksViz::renderLoop(this=0x00007ffeefbfe220) at OpticksViz.cc:474
        frame #16: 0x00000001001c6132 libOGLRap.dylib`OpticksViz::visualize(this=0x00007ffeefbfe220) at OpticksViz.cc:135
        frame #17: 0x0000000100015328 OKX4Test`main(argc=1, argv=0x00007ffeefbfe9c0) at OKX4Test.cc:80
        frame #18: 0x00007fff55eb1015 libdyld.dylib`start + 1
    (lldb) 

In the trace launch::

    (lldb) f 14
    frame #14: 0x00000001001c7de1 libOGLRap.dylib`OpticksViz::render(this=0x00007ffeefbfe220) at OpticksViz.cc:432
       429 	
       430 	    if(m_scene->isRaytracedRender() || m_scene->isCompositeRender()) 
       431 	    {
    -> 432 	        if(m_external_renderer) m_external_renderer->render();
       433 	    }
       434 	
       435 	    m_scene->render();
    (lldb) f 13
    frame #13: 0x00000001001318d5 libOpticksGL.dylib`OKGLTracer::render(this=0x0000000123fc1f70) at OKGLTracer.cc:165
       162 	        {
       163 	            unsigned int scale = m_interactor->getOptiXResolutionScale() ;
       164 	            m_otracer->setResolutionScale(scale) ;
    -> 165 	            m_otracer->trace_();
       166 	            m_oframe->push_PBO_to_Texture();
       167 	
       168 	/*
    (lldb) 



FIXED : X4 Conversion missing scintillators causing crash in OScintillatorLib::convert
-------------------------------------------------------------------------------------------

* looks to be from lack of GGeo::addRaw in X4 
* fixed by rejig of material handling, moving stuff from GGeo into GMaterialLib 

After fix::

    2018-06-24 21:08:37.846 INFO  [26433846] [*X4PhysicalVolume::convertNode@467] convertNode  ndIdx 12000 soIdx   224 lvIdx   218 materialIdx    15 soName out_cross_rib0xc20ec60
    2018-06-24 21:08:37.876 INFO  [26433846] [X4PhysicalVolume::convertStructure@369]  convertStructure END  Sc  nodes:12230 meshes: 249
    2018-06-24 21:08:37.877 ERROR [26433846] [GMaterialLib::getRawMaterialsWithProperties@884] GMaterialLib::getRawMaterialsWithProperties SLOWCOMPONENT,FASTCOMPONENT,REEMISSIONPROB m_materials_raw.size()  36
    2018-06-24 21:08:37.877 INFO  [26433846] [GGeo::prepareScintillatorLib@1144] GGeo::prepareScintillatorLib found 2 scintillator materials  
    2018-06-24 21:08:37.877 INFO  [26433846] [*GScintillatorLib::createBuffer@109] GScintillatorLib::createBuffer  ni 2 nj 4096 nk 1

Issue::

    2018-06-24 17:59:32.143 INFO  [26324769] [*X4PhysicalVolume::convertNode@467] convertNode  ndIdx 12000 soIdx   224 lvIdx   218 materialIdx    15 soName out_cross_rib0xc20ec60
    2018-06-24 17:59:32.173 INFO  [26324769] [X4PhysicalVolume::convertStructure@369]  convertStructure END  Sc  nodes:12230 meshes: 249
    2018-06-24 17:59:32.173 ERROR [26324769] [GGeo::getRawMaterialsWithProperties@1323] GGeo::getRawMaterialsWithProperties SLOWCOMPONENT,FASTCOMPONENT,REEMISSIONPROB m_materials_raw.size()  0
    2018-06-24 17:59:32.173 ERROR [26324769] [GGeo::prepareScintillatorLib@1173] GGeo::prepareScintillatorLib found no scintillator materials  
    2018-06-24 17:59:32.173 INFO  [26324769] [*GSourceLib::createBuffer@88] GSourceLib::createBuffer adding standard source 
    2018-06-24 17:59:32.174 INFO  [26324769] [GPropertyLib::close@418] GPropertyLib::close type GSourceLib buf 1,1024,1

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x78)
      * frame #0: 0x000000010b711a03 libNPY.dylib`NPYBase::getShape(unsigned int) const [inlined] std::__1::vector<int, std::__1::allocator<int> >::size(this=0x0000000000000070 size=1) const at vector:632
        frame #1: 0x000000010b711a03 libNPY.dylib`NPYBase::getShape(this=0x0000000000000000, n=0) const at NPYBase.cpp:235
        frame #2: 0x00000001004ddc19 libOptiXRap.dylib`OScintillatorLib::convert(this=0x000000012c88ed40, slice="0:1") at OScintillatorLib.cc:20
        frame #3: 0x00000001004e27e7 libOptiXRap.dylib`OScene::init(this=0x000000011f05a940) at OScene.cc:148
        frame #4: 0x00000001004e1794 libOptiXRap.dylib`OScene::OScene(this=0x000000011f05a940, hub=0x00007ffeefbfe2e0) at OScene.cc:78
        frame #5: 0x00000001004e31fd libOptiXRap.dylib`OScene::OScene(this=0x000000011f05a940, hub=0x00007ffeefbfe2e0) at OScene.cc:77
        frame #6: 0x0000000100407d1e libOKOP.dylib`OpEngine::OpEngine(this=0x000000011f062550, hub=0x00007ffeefbfe2e0) at OpEngine.cc:44
        frame #7: 0x000000010040820d libOKOP.dylib`OpEngine::OpEngine(this=0x000000011f062550, hub=0x00007ffeefbfe2e0) at OpEngine.cc:52
        frame #8: 0x000000010010a5c6 libOK.dylib`OKPropagator::OKPropagator(this=0x00007ffeefbfe1e0, hub=0x00007ffeefbfe2e0, idx=0x00007ffeefbfe2c8, viz=0x00007ffeefbfe220) at OKPropagator.cc:50
        frame #9: 0x000000010010a72d libOK.dylib`OKPropagator::OKPropagator(this=0x00007ffeefbfe1e0, hub=0x00007ffeefbfe2e0, idx=0x00007ffeefbfe2c8, viz=0x00007ffeefbfe220) at OKPropagator.cc:54
        frame #10: 0x0000000100015317 OKX4Test`main(argc=1, argv=0x00007ffeefbfe9c0) at OKX4Test.cc:78
        frame #11: 0x00007fff55eb1015 libdyld.dylib`start + 1
    (lldb) exit
    Quitting LLDB will kill one or more processes. Do you really want to proceed: [Y/n] 
    epsilon:okg4 blyth$ 



FIXED : OKPropagator instanciation fails for lack of source buffer
--------------------------------------------------------------------

* fixed by addition of GGeo::prepareSourceLib to GGeo::prepare that closes the sourcelib


::

    2018-06-24 17:26:12.546 INFO  [26242428] [OScene::init@105] OScene::init START
    2018-06-24 17:26:12.706 INFO  [26242428] [OScene::init@130] OScene::init ggeobase identifier : GGeo
    2018-06-24 17:26:12.706 WARN  [26242428] [OColors::convert@30] OColors::convert SKIP no composite color buffer 
    Assertion failed: (buf && "OSourceLib::makeSourceTexture NULL buffer, try updating geocache first: ggv -G  ? "), function makeSourceTexture, file /Users/blyth/opticks-cmake-overhaul/optixrap/OSourceLib.cc, line 26.
    Process 39493 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff56001b6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff56001b6e <+10>: jae    0x7fff56001b78            ; <+20>
        0x7fff56001b70 <+12>: movq   %rax, %rdi
        0x7fff56001b73 <+15>: jmp    0x7fff55ff8b00            ; cerror_nocancel
        0x7fff56001b78 <+20>: retq   
    Target 0: (OKX4Test) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff56001b6e libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff561cc080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff55f5d1ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff55f251ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001004dee74 libOptiXRap.dylib`OSourceLib::makeSourceTexture(this=0x000000011c2e4960, buf=0x0000000000000000) at OSourceLib.cc:26
        frame #5: 0x00000001004dede5 libOptiXRap.dylib`OSourceLib::convert(this=0x000000011c2e4960) at OSourceLib.cc:19
        frame #6: 0x00000001004e2607 libOptiXRap.dylib`OScene::init(this=0x0000000128701960) at OScene.cc:142
        frame #7: 0x00000001004e1794 libOptiXRap.dylib`OScene::OScene(this=0x0000000128701960, hub=0x00007ffeefbfe2e0) at OScene.cc:78
        frame #8: 0x00000001004e31fd libOptiXRap.dylib`OScene::OScene(this=0x0000000128701960, hub=0x00007ffeefbfe2e0) at OScene.cc:77
        frame #9: 0x0000000100407d1e libOKOP.dylib`OpEngine::OpEngine(this=0x00000001287018e0, hub=0x00007ffeefbfe2e0) at OpEngine.cc:44
        frame #10: 0x000000010040820d libOKOP.dylib`OpEngine::OpEngine(this=0x00000001287018e0, hub=0x00007ffeefbfe2e0) at OpEngine.cc:52
        frame #11: 0x000000010010a5c6 libOK.dylib`OKPropagator::OKPropagator(this=0x00007ffeefbfe1e0, hub=0x00007ffeefbfe2e0, idx=0x00007ffeefbfe2c8, viz=0x00007ffeefbfe220) at OKPropagator.cc:50
        frame #12: 0x000000010010a72d libOK.dylib`OKPropagator::OKPropagator(this=0x00007ffeefbfe1e0, hub=0x00007ffeefbfe2e0, idx=0x00007ffeefbfe2c8, viz=0x00007ffeefbfe220) at OKPropagator.cc:54
        frame #13: 0x0000000100015317 OKX4Test`main(argc=1, argv=0x00007ffeefbfe9c0) at OKX4Test.cc:78
        frame #14: 0x00007fff55eb1015 libdyld.dylib`start + 1
    (lldb) f 4
    frame #4: 0x00000001004dee74 libOptiXRap.dylib`OSourceLib::makeSourceTexture(this=0x000000011c2e4960, buf=0x0000000000000000) at OSourceLib.cc:26
       23  	{
       24  	   // this is fragile, often getting memory errors
       25  	
    -> 26  	    assert(buf && "OSourceLib::makeSourceTexture NULL buffer, try updating geocache first: ggv -G  ? " );
       27  	
       28  	    unsigned int ni = buf->getShape(0);
       29  	    unsigned int nj = buf->getShape(1);
    (lldb) f 5
    frame #5: 0x00000001004dede5 libOptiXRap.dylib`OSourceLib::convert(this=0x000000011c2e4960) at OSourceLib.cc:19
       16  	{
       17  	    LOG(debug) << "OSourceLib::convert" ;
       18  	    NPY<float>* buf = m_lib->getBuffer();
    -> 19  	    makeSourceTexture(buf);
       20  	}
       21  	
       22  	void OSourceLib::makeSourceTexture(NPY<float>* buf)
    (lldb) p m_lib
    (GSourceLib *) $0 = 0x000000011554def0
    (lldb) 




FIXED : Switching to raytrace render with O crashes in Renderer::render
--------------------------------------------------------------------------

* fixed by adding inhibition of raytrace rendering 
  when the interop setup in OKGLTracer has not been done  



The raytrace rendering relies on GPU side interop between OptiX and OpenGL 
which is coordinated by okgl.OKGLTracer.  If there is no instance of 
that booted up and called every frame, you get the crash.

* hmm how to detect that and prevent the O option from doing anything ?


* to do this is a bit of a dependency conundrum, as only the packages above OKGL
  can check on that the instance is around OKGLTracer::GetInstance() 
  but the rendering style control and the crash is back down in OGLRap :
  perhaps just a setEnableRayTracing on Scene that needs to be called
  from on high


::

    050          OKCORE :          optickscore :          OpticksCore : NPY  
     60            GGEO :                 ggeo :                 GGeo : OpticksCore  
     65              X4 :                extg4 :                ExtG4 : G4 GGeo YoctoGLRap  
     70          ASIRAP :            assimprap :            AssimpRap : OpticksAssimp GGeo  
     80         MESHRAP :          openmeshrap :          OpenMeshRap : GGeo OpticksCore  
     90           OKGEO :           opticksgeo :           OpticksGeo : OpticksCore AssimpRap OpenMeshRap  
    100         CUDARAP :              cudarap :              CUDARap : OKConf SysRap OpticksCUDA  
    110           THRAP :            thrustrap :            ThrustRap : OKConf OpticksCore CUDARap  
    120           OXRAP :             optixrap :             OptiXRap : OKConf OptiX OpticksGeo ThrustRap  
    130            OKOP :                 okop :                 OKOP : OKConf OptiXRap  
    140          OGLRAP :               oglrap :               OGLRap : ImGui OpticksGLEW OpticksGLFW OpticksGeo  
    150            OKGL :            opticksgl :            OpticksGL : OGLRap OKOP  
    160              OK :                   ok :                   OK : OpticksGL  
    170            CFG4 :                 cfg4 :                 CFG4 : G4 ExtG4 OpticksXercesC OpticksGeo  
    180            OKG4 :                 okg4 :                 OKG4 : OK CFG4  
    190            G4OK :                 g4ok :                 G4OK : CFG4 OKConf OKOP G4DAE  





::

   lldb OKX4Test


    018-06-24 15:35:26.738 INFO  [25996296] [Interactor::key_pressed@409] Interactor::key_pressed O nextRenderStyle 
    Process 20950 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x0)
        frame #0: 0x00000001001be65a libOGLRap.dylib`Renderer::render(this=0x000000011c563f70) at Renderer.cc:638
       635 	        else
       636 	        {
       637 	            //LOG(info) << "glDrawElements " << draw.desc() ;  
    -> 638 	            glDrawElements( draw.mode, draw.count, draw.type,  draw.indices ) ;
       639 	        }
       640 	    }
       641 	
    Target 0: (OKX4Test) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x0)
      * frame #0: 0x00000001001be65a libOGLRap.dylib`Renderer::render(this=0x000000011c563f70) at Renderer.cc:638
        frame #1: 0x00000001001abe32 libOGLRap.dylib`Scene::render(this=0x000000011c55f8c0) at Scene.cc:883
        frame #2: 0x00000001001c5fb3 libOGLRap.dylib`OpticksViz::render(this=0x00007ffeefbfe0e0) at OpticksViz.cc:435
        frame #3: 0x00000001001c4bb2 libOGLRap.dylib`OpticksViz::renderLoop(this=0x00007ffeefbfe0e0) at OpticksViz.cc:474
        frame #4: 0x00000001001c42f2 libOGLRap.dylib`OpticksViz::visualize(this=0x00007ffeefbfe0e0) at OpticksViz.cc:135
        frame #5: 0x0000000100015342 OKX4Test`main(argc=1, argv=0x00007ffeefbfe880) at OKX4Test.cc:77
        frame #6: 0x00007fff55eb1015 libdyld.dylib`start + 1
    (lldb) p this
    (Renderer *) $0 = 0x000000011c563f70
    (lldb) p *this
    (Renderer) $1 = {
      RendererBase = {
        m_shader = 0x000000011c5640d0
        m_program = -1
        m_verbosity = 0
        m_shaderdir = 0x000000011c561c40 "/usr/local/opticks-cmake-overhaul/gl"
        m_shadertag = 0x000000011c563070 "tex"
        m_incl_path = 0x000000011c563ce0 "/usr/local/opticks-cmake-overhaul/gl"
      }
      m_vao = ([0] = 1936484142, [1] = 108, [2] = 0)
      m_vao_all = 0
      m_draw = {
        [0] = 0x0000000000000000
        [1] = 0x0000000000000000
        [2] = 0x0000000000000023
      }
      m_draw_0 = 0
      m_draw_1 = 1
      m_lod_counts = ([0] = 0, [1] = 0, [2] = 0)
      m_vbuf = 0x0000000000000000


::

    0966 void Scene::nextRenderStyle(unsigned int modifiers)  // O:key cycling: Projective, Raytraced, Composite 
     967 {
     968     bool nudge = modifiers & OpticksConst::e_shift ;
     969     if(nudge)
     970     {
     971         m_composition->setChanged(true) ;
     972         return ;
     973     }
     974 
     975     int next = (m_render_style + 1) % NUM_RENDER_STYLE ;
     976     m_render_style = (RenderStyle_t)next ;
     977     applyRenderStyle();
     978 
     979     m_composition->setChanged(true) ; // trying to avoid the need for shift-O nudging 
     980 }
     981 
     982 
     983 
     984 
     985 bool Scene::isProjectiveRender() const
     986 {
     987    return m_render_style == R_PROJECTIVE ;
     988 }
     989 bool Scene::isRaytracedRender() const
     990 {
     991    return m_render_style == R_RAYTRACED ;
     992 }
     993 bool Scene::isCompositeRender() const
     994 {
     995    return m_render_style == R_COMPOSITE ;
     996 }
     997 
     998 void Scene::applyRenderStyle()
     999 {
    1000     // nothing to do, style is honoured by  Scene::render
    1001 
    1002 
    1003 }





FIXED : Targetting difference yields a blank screen for OKX4Test
------------------------------------------------------------------

* OpticksQuery selection was not being applied by X4PhysicalVolume, so the
  merged mesh was an enormous one (from overlarge world volume)


OKTest::

    2018-06-23 23:28:00.106 INFO  [25695381] [*OpticksHub::getGGeoBasePrimary@726] OpticksHub::getGGeoBasePrimary analytic switch   m_gltf 0 ggb GGeo
       0 **                                    World0xc15cfc0         ce-16520.000 -802110.000 -7125.000 7710.562 
       1       __dd__Geometry__Sites__lvNearSiteRock0xc030350         ce-16520.000 -802110.000 3892.900 34569.875 
       2        __dd__Geometry__Sites__lvNearHallTop0xc136890         ce-12840.846 -806876.250 5389.855 22545.562 
       3   __dd__Geometry__PoolDetails__lvNearTopCover0xc137060         ce-16519.969 -802109.875 -2088.000 7800.906 
       4               __dd__Geometry__RPC__lvRPCMod0xbf54e60         ce-11612.387 -799007.250 683.900 1509.703 
       5              __dd__Geometry__RPC__lvRPCFoam0xc032c88         ce-11611.265 -799018.375 683.900 1455.636 
       6         __dd__Geometry__RPC__lvRPCBarCham140xbf4c6a0         ce-11611.265 -799018.375 669.900 1448.750 
       7          __dd__Geometry__RPC__lvRPCGasgap140xbf98ae0         ce-11611.265 -799018.375 669.900 1434.939 
       8             __dd__Geometry__RPC__lvRPCStrip0xc2213c0         ce-11124.670 -799787.375 669.900 948.345 
       9             __dd__Geometry__RPC__lvRPCStrip0xc2213c0         ce-11263.697 -799567.625 669.900 948.345 
      10             __dd__Geometry__RPC__lvRPCStrip0xc2213c0         ce-11402.724 -799347.938 669.900 948.345 
      11             __dd__Geometry__RPC__lvRPCStrip0xc2213c0         ce-11541.751 -799128.250 669.900 948.345 
      12             __dd__Geometry__RPC__lvRPCStrip0xc2213c0         ce-11680.778 -798908.500 669.900 948.345 
      13             __dd__Geometry__RPC__lvRPCStrip0xc2213c0         ce-11819.806 -798688.812 669.900 948.345 
      14             __dd__Geometry__RPC__lvRPCStrip0xc2213c0         ce-11958.832 -798469.125 669.900 948.345 
      15             __dd__Geometry__RPC__lvRPCStrip0xc2213c0         ce-12097.859 -798249.375 669.900 948.345 
      16         __dd__Geometry__RPC__lvRPCBarCham140xbf4c6a0         ce-11611.265 -799018.375 707.900 1448.750 
      17          __dd__Geometry__RPC__lvRPCGasgap140xbf98ae0         ce-11611.265 -799018.375 707.900 1434.939 
      18             __dd__Geometry__RPC__lvRPCStrip0xc2213c0         ce-11124.670 -799787.375 707.900 948.345 
      19             __dd__Geometry__RPC__lvRPCStrip0xc2213c0         ce-11263.697 -799567.625 707.900 948.345 
    2018-06-23 23:28:00.106 FATAL [25695381] [OpticksAim::setTarget@119] OpticksAim::setTarget  based on CenterExtent from m_mesh0  target 0 aim 1 ce -16520.0000,-802110.0000,-7125.0000,7710.5625
    2018-06-23 23:28:00.106 INFO  [25695381] [Composition::setCenterExtent@1010] Composition::setCenterExtent ce -16520.0000,-802110.0000,-7125.0000,7710.5625
    2018-06-23 23:28:00.106 INFO  [25695381] [SLog::operator@20] OpticksViz::OpticksViz  DONE


OKX4Test::

    2018-06-23 23:31:04.004 INFO  [25697900] [OpticksAim::setupCompositionTargetting@92] OpticksAim::setupCompositionTargetting deferred_target 0 cmdline_target 0
    2018-06-23 23:31:04.004 INFO  [25697900] [OpticksHub::dumpVolumes@887] OpticksHub::dumpVolumes OpticksAim::setTarget num_volumes 12230
    2018-06-23 23:31:04.005 INFO  [25697900] [*OpticksHub::getGGeoBasePrimary@726] OpticksHub::getGGeoBasePrimary analytic switch   m_gltf 0 ggb GGeo
       0 **                                    World0xc15cfc0         ce  0.000   0.000   0.000 2400000.000 
       1           /dd/Geometry/Sites/lvNearSiteRock0xc030350         ce-16520.000 -802110.000 3892.925 34569.875 
       2            /dd/Geometry/Sites/lvNearHallTop0xc136890         ce-12841.452 -806876.000 5390.000 22545.344 
       3     /dd/Geometry/PoolDetails/lvNearTopCover0xc137060         ce-16520.098 -802110.000 -2088.000 7801.031 
       4                   /dd/Geometry/RPC/lvRPCMod0xbf54e60         ce-11612.390 -799007.250 683.903 1509.703 
       5                  /dd/Geometry/RPC/lvRPCFoam0xc032c88         ce-11611.268 -799018.375 683.903 1455.636 
       6             /dd/Geometry/RPC/lvRPCBarCham140xbf4c6a0         ce-11611.268 -799018.375 669.903 1448.750 
       7              /dd/Geometry/RPC/lvRPCGasgap140xbf98ae0         ce-11611.268 -799018.375 669.903 1434.939 
       8                 /dd/Geometry/RPC/lvRPCStrip0xc2213c0         ce-11124.673 -799787.375 669.903 948.345 
       9                 /dd/Geometry/RPC/lvRPCStrip0xc2213c0         ce-11263.700 -799567.625 669.903 948.345 
      10                 /dd/Geometry/RPC/lvRPCStrip0xc2213c0         ce-11402.727 -799347.938 669.903 948.345 
      11                 /dd/Geometry/RPC/lvRPCStrip0xc2213c0         ce-11541.754 -799128.250 669.903 948.345 
      12                 /dd/Geometry/RPC/lvRPCStrip0xc2213c0         ce-11680.781 -798908.500 669.903 948.345 
      13                 /dd/Geometry/RPC/lvRPCStrip0xc2213c0         ce-11819.809 -798688.812 669.903 948.345 
      14                 /dd/Geometry/RPC/lvRPCStrip0xc2213c0         ce-11958.835 -798469.125 669.903 948.345 
      15                 /dd/Geometry/RPC/lvRPCStrip0xc2213c0         ce-12097.862 -798249.375 669.903 948.345 
      16             /dd/Geometry/RPC/lvRPCBarCham140xbf4c6a0         ce-11611.268 -799018.375 707.903 1448.750 
      17              /dd/Geometry/RPC/lvRPCGasgap140xbf98ae0         ce-11611.268 -799018.375 707.903 1434.939 
      18                 /dd/Geometry/RPC/lvRPCStrip0xc2213c0         ce-11124.673 -799787.375 707.903 948.345 
      19                 /dd/Geometry/RPC/lvRPCStrip0xc2213c0         ce-11263.700 -799567.625 707.903 948.345 
    2018-06-23 23:31:04.005 FATAL [25697900] [OpticksAim::setTarget@119] OpticksAim::setTarget  based on CenterExtent from m_mesh0  target 0 aim 1 ce 0.0000,0.0000,0.0000,2400000.0000
    2018-06-23 23:31:04.005 INFO  [25697900] [Composition::setCenterExtent@1010] Composition::setCenterExtent ce 0.0000,0.0000,0.0000,2400000.0000
    2018-06-23 23:31:04.005 INFO  [25697900] [SLog::operator@20] OpticksViz::OpticksViz  DONE
    2018-06-23 23:31:04.005 INFO  [25697900] [Bookmarks::create@249] Bookmarks::create : persisting state to slot 0



Geocache matching : its going to take a while ... 
-------------------------------------------------------

* to get a match will take at least a week of detailed work : not the best use of time at the moment

* perhaps : try to push ahead and see if can run from the directly converted GGeo, eg 

  * OGLRap render
  * ray trace
  * OptiX sim 


Basically this means modifying some tests to boot from the direct GGeo

* actually the direct GGeo is from the CGDMLDetector load ... 


Three Solids X4Mesh skipped still 
------------------------------------

::

    443      std::vector<unsigned> skips = {27, 29, 33 };
    444 
    445      if(mh->csgnode == NULL)
    446      {
    447          mh->csgnode = X4Solid::Convert(solid) ;  // soIdx 33 giving analytic problems too 
    448 
    449          bool placeholder = std::find( skips.begin(), skips.end(), nd->soIdx ) != skips.end()  ;
    450 
    451          mh->mesh = placeholder ? X4Mesh::Placeholder(solid) : X4Mesh::Convert(solid) ;
    452 


PVNames / LVNames
--------------------

Some name fixup done following the GDML load ?  

::

    epsilon:src blyth$ geocache-;geocache-diff-lv | head -10

    ======== GNodeLib/LVNames.txt 

    World0xc15cfc0							World0xc15cfc0
    __dd__Geometry__Sites__lvNearSiteRock0xc030350		      |	/dd/Geometry/Sites/lvNearSiteRock0xc030350
    __dd__Geometry__Sites__lvNearHallTop0xc136890		      |	/dd/Geometry/Sites/lvNearHallTop0xc136890
    __dd__Geometry__PoolDetails__lvNearTopCover0xc137060	      |	/dd/Geometry/PoolDetails/lvNearTopCover0xc137060
    __dd__Geometry__RPC__lvRPCMod0xbf54e60			      |	/dd/Geometry/RPC/lvRPCMod0xbf54e60
    __dd__Geometry__RPC__lvRPCFoam0xc032c88			      |	/dd/Geometry/RPC/lvRPCFoam0xc032c88
    __dd__Geometry__RPC__lvRPCBarCham140xbf4c6a0		      |	/dd/Geometry/RPC/lvRPCBarCham140xbf4c6a0
    epsilon:src blyth$ 


Name in the GDML is path like, but is converted to XML friendly form before reaching geocache::


    30919     <volume name="/dd/Geometry/Sites/lvNearSiteRock0xc030350">
    30920       <materialref ref="/dd/Materials/Rock0xc0300c8"/>
    30921       <solidref ref="near_rock0xc04ba08"/>
    30922       <physvol name="/dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop0xbf89820">
    30923         <volumeref ref="/dd/Geometry/Sites/lvNearHallTop0xc136890"/>
    30924         <position name="/dd/Geometry/Sites/lvNearSiteRock#pvNearHallTop0xbf89820_pos" unit="mm" x="2500" y="-500" z="7500"/>
    30925       </physvol>
    30926       <physvol name="/dd/Geometry/Sites/lvNearSiteRock#pvNearHallBot0xcd2fa58">
    30927         <volumeref ref="/dd/Geometry/Sites/lvNearHallBot0xbf89c60"/>
    30928         <position name="/dd/Geometry/Sites/lvNearSiteRock#pvNearHallBot0xcd2fa58_pos" unit="mm" x="0" y="0" z="-5150"/>
    30929       </physvol>
    30930     </volume>





No surfaces listed ? UNDERSTOOD
-------------------------------------

Am testing from an old DYB GDML loaded geometry (which lacks surfaces).  It also 
lacked MPT : which are fixed up from the G4DAE in cfg4.CGDMLDetector ?

* how to proceed ? do some more fixup ?

::

    2018-06-23 20:29:00.568 ERROR [25544667] [X4LogicalBorderSurfaceTable::init@32]  NumberOfBorderSurfaces 0
    2018-06-23 20:29:00.568 ERROR [25544667] [X4LogicalSkinSurfaceTable::init@32]  NumberOfSkinSurfaces 0
    2018-06-23 20:29:00.568 INFO  [25544667] [X4PhysicalVolume::convertSurfaces@175] convertSurfaces num_lbs 0 num_sks 0
    2018-06-23 20:29:00.568 INFO  [25544667] [GPropertyLib::close@417] GPropertyLib::close type GSurfaceLib buf 4,2,39,4

::

    In [5]: aa.shape
    Out[5]: (48, 2, 39, 4)

    In [6]: bb.shape
    Out[6]: (4, 2, 39, 4)

::

    epsilon:ana blyth$ cat /usr/local/opticks-cmake-overhaul/geocache/CX4GDMLTest_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1/GItemList/GSurfaceLib.txt 
    perfectDetectSurface
    perfectAbsorbSurface
    perfectSpecularSurface
    perfectDiffuseSurface
    epsilon:ana blyth$ 



Comparing geocache : some large differences in groupvel ? UNDERSTOOD
------------------------------------------------------------------------

Huh : the old geocache material groupvel always 300, but the 
new one is varying.  Was that a postcache fixup ? 

* Ah-ha : the fixup was done postcache (GMaterialLib::postLoadFromCache) 
  SO THE 300. IN THE OLD GEOCACHE ARE UNDERSTOOD : DIFFERENCE IS UNDERSTOOD 


::

    055 void GMaterialLib::postLoadFromCache()
     56 {
     ..
     69     bool groupvel = !m_ok->hasOpt("nogroupvel") ;
     70 

    119     if(groupvel)   // unlike the other material changes : this one is ON by default, so long at not swiched off with --nogroupvel
    120     {
    121        bool debug = false ;
    122        replaceGROUPVEL(debug);
    123     }
    124 




::

    In [58]: cat geocache.py 
    #!/usr/bin/env python

    import os, numpy as np

    idp_ = lambda _:os.path.expandvars("$IDPATH/%s" % _ )
    idp2_ = lambda _:os.path.expandvars("$IDPATH2/%s" % _ )


    if __name__ == '__main__':
        aa = np.load(idp_("GMaterialLib/GMaterialLib.npy"))
        bb = np.load(idp2_("GMaterialLib/GMaterialLib.npy"))
        assert aa.shape == bb.shape
        print aa.shape

        for i in range(len(aa)):
            a = aa[i]  
            b = bb[i]  
            assert len(a) == 2 
            assert len(b) == 2 

            g0 = a[0] - b[0] 
            g1 = a[1] - b[1] 

            assert g0.shape == g1.shape

            print i, g0.shape, "g0max: ", np.max(g0), "g1max: ", np.max(g1)




::

    In [51]: aa[:,1,:,0]
    Out[51]: 
    array([[300., 300., 300., ..., 300., 300., 300.],
           [300., 300., 300., ..., 300., 300., 300.],
           [300., 300., 300., ..., 300., 300., 300.],
           ...,
           [300., 300., 300., ..., 300., 300., 300.],
           [300., 300., 300., ..., 300., 300., 300.],
           [300., 300., 300., ..., 300., 300., 300.]], dtype=float32)

    In [52]: aa[:,1,:,0].shape
    Out[52]: (38, 39)

    In [53]: aa[:,1,:,0].min()
    Out[53]: 300.0

    In [54]: aa[:,1,:,0].max()
    Out[54]: 300.0

    In [55]: bb[:,1,:,0]
    Out[55]: 
    array([[206.2414, 206.2414, 206.2414, ..., 200.9359, 201.9052, 202.8228],
           [206.2414, 206.2414, 206.2414, ..., 200.9359, 201.9052, 202.8228],
           [205.0564, 205.0564, 205.0564, ..., 199.8321, 200.6891, 201.5005],
           ...,
           [299.7924, 299.7924, 299.7924, ..., 299.7924, 299.7924, 299.7924],
           [299.7924, 299.7924, 299.7924, ..., 299.7924, 299.7924, 299.7924],
           [300.    , 300.    , 300.    , ..., 300.    , 300.    , 300.    ]], dtype=float32)

    In [56]: bb[:,1,:,0].min()
    Out[56]: 118.98735

    In [57]: bb[:,1,:,0].max()
    Out[57]: 300.0




::

    In [22]: run geocache.py 
    (38, 2, 39, 4)
    0 (39, 4) g0max:  0.015625 g1max:  181.01265
    1 (39, 4) g0max:  0.015625 g1max:  181.01265
    2 (39, 4) g0max:  0.015625 g1max:  180.42665
    3 (39, 4) g0max:  0.015625 g1max:  178.10599
    4 (39, 4) g0max:  0.00024414062 g1max:  94.38103
    5 (39, 4) g0max:  0.005859375 g1max:  93.02899
    6 (39, 4) g0max:  0.005859375 g1max:  93.02899
    7 (39, 4) g0max:  0.005859375 g1max:  93.02899
    8 (39, 4) g0max:  0.005859375 g1max:  93.02899
    9 (39, 4) g0max:  0.0 g1max:  0.20755005
    10 (39, 4) g0max:  0.0 g1max:  0.20755005
    11 (39, 4) g0max:  0.0 g1max:  0.20755005
    12 (39, 4) g0max:  0.0 g1max:  0.20755005
    13 (39, 4) g0max:  0.00024414062 g1max:  94.38103
    14 (39, 4) g0max:  0.0 g1max:  0.28848267
    15 (39, 4) g0max:  0.0 g1max:  0.0
    16 (39, 4) g0max:  0.0 g1max:  0.20755005
    17 (39, 4) g0max:  0.0 g1max:  0.20755005
    18 (39, 4) g0max:  0.0 g1max:  0.20755005
    19 (39, 4) g0max:  0.0 g1max:  0.20755005
    20 (39, 4) g0max:  0.0 g1max:  0.20755005
    21 (39, 4) g0max:  0.0 g1max:  0.31243896
    22 (39, 4) g0max:  0.0 g1max:  0.20755005
    23 (39, 4) g0max:  0.0 g1max:  0.20755005
    24 (39, 4) g0max:  0.0 g1max:  0.20755005
    25 (39, 4) g0max:  0.0 g1max:  0.20755005
    26 (39, 4) g0max:  0.0 g1max:  0.20755005
    27 (39, 4) g0max:  0.0 g1max:  0.20755005
    28 (39, 4) g0max:  0.015625 g1max:  180.42665
    29 (39, 4) g0max:  0.0 g1max:  0.20755005
    30 (39, 4) g0max:  0.0 g1max:  0.20755005
    31 (39, 4) g0max:  0.0 g1max:  0.20755005
    32 (39, 4) g0max:  0.0 g1max:  0.20755005
    33 (39, 4) g0max:  0.0 g1max:  0.20755005
    34 (39, 4) g0max:  0.0 g1max:  0.20755005
    35 (39, 4) g0max:  0.0 g1max:  0.20755005
    36 (39, 4) g0max:  0.0 g1max:  0.20755005
    37 (39, 4) g0max:  0.0 g1max:  0.0




FIXED : Comparing geocache : material lib ordering and test materials
---------------------------------------------------------------------------

* sort material order

  * sorting done by GPropertyLib::close, based on Order from m_attrnames 

::

    338 std::map<std::string, unsigned int>& GPropertyLib::getOrder()
    339 {
    340     return m_attrnames->getOrder() ;
    341 }


GPropertyLib::init loads the prefs including the order::

    318     m_attrnames = new OpticksAttrSeq(m_ok, m_type);
    319     m_attrnames->loadPrefs(); // color.json, abbrev.json and order.json 
    320     LOG(debug) << "GPropertyLib::init loadPrefs-DONE " ;

::

    OpticksResourceTest:

                     detector_base :  Y :      /usr/local/opticks/opticksdata/export/DayaBay


    epsilon:issues blyth$ ll /usr/local/opticks/opticksdata/export/DayaBay/GMaterialLib/
    -rw-r--r--  1 blyth  staff  612 Apr  4 14:26 abbrev.json
    -rw-r--r--  1 blyth  staff  660 Apr  4 14:26 color.json
    -rw-r--r--  1 blyth  staff  795 Apr  4 14:26 order.json


::

   OPTICKS_KEY=CX4GDMLTest.X4PhysicalVolume.World0xc15cfc0_PV.828722902b5e94dab05ac248329ffebe OpticksResourceTest 


Kludge symbolic link to try to access the prefs with the g4live running::

    epsilon:~ blyth$ cd /usr/local/opticks-cmake-overhaul/opticksdata/export/
    epsilon:export blyth$ ln -s DayaBay CX4GDMLTest


* add test materials

::

    export IDPATH2=/usr/local/opticks-cmake-overhaul/geocache/CX4GDMLTest_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1

    epsilon:ana blyth$ python geocache.py 
    (38, 2, 39, 4)
    (36, 2, 39, 4)

::

    epsilon:1 blyth$ head -5 $IDPATH/GItemList/GMaterialLib.txt 
    GdDopedLS
    LiquidScintillator
    Acrylic
    MineralOil
    Bialkali
    epsilon:1 blyth$ head -5 $IDPATH2/GItemList/GMaterialLib.txt 
    PPE
    MixGas
    Air
    Bakelite
    Foam




FIXED : material names with slashes mess up boundary spec 
------------------------------------------------------------

* fixed using basenames

cfg4-;cfg4-c;om-;TEST=CX4GDMLTest om-d::

    2018-06-23 16:30:36.316 INFO  [25301620] [GParts::close@802] GParts::close START  verbosity 0
    2018-06-23 16:30:36.316 FATAL [25301620] [GBnd::init@27] GBnd::init bad boundary spec, expecting 4 elements spec /dd/Materials/Vacuum////dd/Materials/Vacuum nelem 10
    Assertion failed: (nelem == 4), function init, file /Users/blyth/opticks-cmake-overhaul/ggeo/GBnd.cc, line 34.
    Process 19616 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff56001b6e libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff56001b6e <+10>: jae    0x7fff56001b78            ; <+20>
        0x7fff56001b70 <+12>: movq   %rax, %rdi
        0x7fff56001b73 <+15>: jmp    0x7fff55ff8b00            ; cerror_nocancel
        0x7fff56001b78 <+20>: retq   
    Target 0: (CX4GDMLTest) stopped.
    (lldb) 




FIXED : Slow convert due to CSG node nudger running at node(not mesh) level ?
-------------------------------------------------------------------------------- 

* moving the nudging to mesh level, gives drastic speedup : now DYB near
  conversion from G4 model to Opticks GGeo and writes out glTF in 5 seconds.

* looks like the slow convert, was related to not having the displacements 
  done already, nevertheless : if this processing can be moved to mesh level 
  ot should be 



X4PhysicalVolume::convertNode::

    434 
    435      Mh* mh = m_sc->get_mesh_for_node( ndIdx );  // node->mesh via soIdx (the local mesh index)
    436 
    437      std::vector<unsigned> skips = {27, 29, 33 };
    438 
    439      if(mh->csg == NULL)
    440      {
    441          //convertSolid(mh, solid);
    442          mh->csg = X4Solid::Convert(solid) ;  // soIdx 33 giving analytic problems too 
    443 
    444          bool placeholder = std::find( skips.begin(), skips.end(), nd->soIdx ) != skips.end()  ;
    445 
    446          mh->mesh = placeholder ? X4Mesh::Placeholder(solid) : X4Mesh::Convert(solid) ;
    447 
    448          mh->vtx = mh->mesh->m_x4src_vtx ;
    449          mh->idx = mh->mesh->m_x4src_idx ;
    450      }
    451 
    452      assert( mh->csg );
    453 
    454      // can this be done at mesh level (ie within the above bracket) ?
    455      // ... would be a big time saving 
    456      // ... see how the boundary is used, also check GParts 
    457 
    458      mh->csg->set_boundary( boundaryName.c_str() ) ;
    459 
    460      NCSG* csg = NCSG::FromNode( mh->csg, NULL );
    461      assert( csg ) ;
    462      assert( csg->isUsedGlobally() );
    463 
    464      const GMesh* mesh = mh->mesh ;   // hmm AssimpGGeo::convertMeshes does deduping/fixing before inclusion in GVolume(GNode) 
    465 
    466      GParts* pts = GParts::make( csg, boundaryName.c_str(), m_verbosity  );  // see GScene::createVolume 
    467 


* WHY does NCSG require nnode to have boundary spec char* ? 

  * Suspect nnode does not need boundary any more ?
  * hmm actually that was probably a convenience for tboolean- passing boundaries in from python,
    so need to keep the capability
  * GParts really needs this spec, as it has a GBndLib to convert the spec 
    into a bndIdx for laying down in buffers


* guess that GParts needs to be at node level, peer with GVolume 






DONE : initial implementation to convert G4DisplacedSolid into nnode CSG 
---------------------------------------------------------------------------

::

     87 G4BooleanSolid::G4BooleanSolid( const G4String& pName,
     88                                       G4VSolid* pSolidA ,
     89                                       G4VSolid* pSolidB ,
     90                                 const G4Transform3D& transform    ) :
     91   G4VSolid(pName), fAreaRatio(0.), fStatistics(1000000), fCubVolEpsilon(0.001),
     92   fAreaAccuracy(-1.), fCubicVolume(0.), fSurfaceArea(0.),
     93   fRebuildPolyhedron(false), fpPolyhedron(0), createdDisplacedSolid(true)
     94 {
     95   fPtrSolidA = pSolidA ;
     96   fPtrSolidB = new G4DisplacedSolid("placedB",pSolidB,transform) ;
     97 }

::

     70 G4DisplacedSolid::G4DisplacedSolid( const G4String& pName,
     71                                           G4VSolid* pSolid ,
     72                                     const G4Transform3D& transform  )
     73   : G4VSolid(pName), fRebuildPolyhedron(false), fpPolyhedron(0)
     74 {
     75   fPtrSolid = pSolid ;
     76   fDirectTransform = new G4AffineTransform(transform.getRotation().inverse(),
     77                                            transform.getTranslation()) ;
     78 
     79   fPtrTransform    = new G4AffineTransform(transform.getRotation().inverse(),
     80                                            transform.getTranslation()) ;
     81   fPtrTransform->Invert() ;
     82 }


g4-gcd::

     152 void G4GDMLWriteSolids::
     153 BooleanWrite(xercesc::DOMElement* solElement,
     154              const G4BooleanSolid* const boolean)
     155 {
     156    G4int displaced=0;
     157 
     158    G4String tag("undefined");
     159    if (dynamic_cast<const G4IntersectionSolid*>(boolean))
     160      { tag = "intersection"; } else
     161    if (dynamic_cast<const G4SubtractionSolid*>(boolean))
     162      { tag = "subtraction"; } else
     163    if (dynamic_cast<const G4UnionSolid*>(boolean))
     164      { tag = "union"; }
     165 
     166    G4VSolid* firstPtr = const_cast<G4VSolid*>(boolean->GetConstituentSolid(0));
     167    G4VSolid* secondPtr = const_cast<G4VSolid*>(boolean->GetConstituentSolid(1));
     168 
     169    G4ThreeVector firstpos,firstrot,pos,rot;
     170 
     171    // Solve possible displacement of referenced solids!
     172    //
     173    while (true)
     174    {
     175       if ( displaced>8 )
     ///                 ... error message ...
     ...
     186       if (G4DisplacedSolid* disp = dynamic_cast<G4DisplacedSolid*>(firstPtr))
     187       {
     188          firstpos += disp->GetObjectTranslation();
     189          firstrot += GetAngles(disp->GetObjectRotation());
     ///
     ///      adding angles ... hmm looks fishy 
     ///
     190          firstPtr = disp->GetConstituentMovedSolid();
     191          displaced++;
     ///
     ///   can understand why you might have one displacement ?
     ///   but how you manage to have 8 displacements ? 
     ///
     192          continue;
     193       }
     194       break;
     195    }
     196    displaced = 0;

     ...
     221    AddSolid(firstPtr);   // At first add the constituent solids!
     222    AddSolid(secondPtr);
     223 
     224    const G4String& name = GenerateName(boolean->GetName(),boolean);
     225    const G4String& firstref = GenerateName(firstPtr->GetName(),firstPtr);
     226    const G4String& secondref = GenerateName(secondPtr->GetName(),secondPtr);
     227 
     228    xercesc::DOMElement* booleanElement = NewElement(tag);
     229    booleanElement->setAttributeNode(NewAttribute("name",name));
     230    xercesc::DOMElement* firstElement = NewElement("first");
     231    firstElement->setAttributeNode(NewAttribute("ref",firstref));
     232    booleanElement->appendChild(firstElement);
     233    xercesc::DOMElement* secondElement = NewElement("second");
     234    secondElement->setAttributeNode(NewAttribute("ref",secondref));
     235    booleanElement->appendChild(secondElement);
     236    solElement->appendChild(booleanElement);
     237      // Add the boolean solid AFTER the constituent solids!
     238 
     239    if ( (std::fabs(pos.x()) > kLinearPrecision)
     240      || (std::fabs(pos.y()) > kLinearPrecision)
     241      || (std::fabs(pos.z()) > kLinearPrecision) )
     242    {
     243      PositionWrite(booleanElement,name+"_pos",pos);
     244    }
     245 
     246    if ( (std::fabs(rot.x()) > kAngularPrecision)
     247      || (std::fabs(rot.y()) > kAngularPrecision)
     248      || (std::fabs(rot.z()) > kAngularPrecision) )
     249    {
     250      RotationWrite(booleanElement,name+"_rot",rot);
     251    }
     252 
     253    if ( (std::fabs(firstpos.x()) > kLinearPrecision)
     254      || (std::fabs(firstpos.y()) > kLinearPrecision)
     255      || (std::fabs(firstpos.z()) > kLinearPrecision) )
     256    {
     257      FirstpositionWrite(booleanElement,name+"_fpos",firstpos);
     258    }
     259 
     260    if ( (std::fabs(firstrot.x()) > kAngularPrecision)
     261      || (std::fabs(firstrot.y()) > kAngularPrecision)
     262      || (std::fabs(firstrot.z()) > kAngularPrecision) )
     263    {
     264      FirstrotationWrite(booleanElement,name+"_frot",firstrot);
     265    }
     266 }


::

     .80 void G4GDMLReadSolids::
      81 BooleanRead(const xercesc::DOMElement* const booleanElement, const BooleanOp op)
      82 {
     ...
     154    G4VSolid* firstSolid = GetSolid(GenerateName(first));
     155    G4VSolid* secondSolid = GetSolid(GenerateName(scnd));
     156 
     157    G4Transform3D transform(GetRotationMatrix(rotation),position);
     158 
     159    if (( (firstrotation.x()!=0.0) || (firstrotation.y()!=0.0)
     160                                   || (firstrotation.z()!=0.0))
     161     || ( (firstposition.x()!=0.0) || (firstposition.y()!=0.0)
     162                                   || (firstposition.z()!=0.0)))
     163    {
     164       G4Transform3D firsttransform(GetRotationMatrix(firstrotation),
     165                                    firstposition);
     166       firstSolid = new G4DisplacedSolid(GenerateName("displaced_"+first),
     167                                         firstSolid, firsttransform);
     168    }
     169 
     170    if (op==UNION)
     171      { new G4UnionSolid(name,firstSolid,secondSolid,transform); } else
     172    if (op==SUBTRACTION)
     173      { new G4SubtractionSolid(name,firstSolid,secondSolid,transform); } else
     174    if (op==INTERSECTION)
     175      { new G4IntersectionSolid(name,firstSolid,secondSolid,transform); }
     176 }

::

    132 G4RotationMatrix
    133 G4GDMLReadDefine::GetRotationMatrix(const G4ThreeVector& angles)
    134 {
    135    G4RotationMatrix rot;
    136 
    137    rot.rotateX(angles.x());
    138    rot.rotateY(angles.y());
    139    rot.rotateZ(angles.z());
    140    rot.rectify();  // Rectify matrix from possible roundoff errors
    141 
    142    return rot;




G4GDMLWriteDefine.hh::

     58     void RotationWrite(xercesc::DOMElement* element,
     59                     const G4String& name, const G4ThreeVector& rot)
     60          { Rotation_vectorWrite(element,"rotation",name,rot); }
     61     void PositionWrite(xercesc::DOMElement* element,
     62                     const G4String& name, const G4ThreeVector& pos)
     63          { Position_vectorWrite(element,"position",name,pos); }
     64     void FirstrotationWrite(xercesc::DOMElement* element,
     65                     const G4String& name, const G4ThreeVector& rot)
     66          { Rotation_vectorWrite(element,"firstrotation",name,rot); }
     67     void FirstpositionWrite(xercesc::DOMElement* element,
     68                     const G4String& name, const G4ThreeVector& pos)
     69          { Position_vectorWrite(element,"firstposition",name,pos); }
     70     void AddPosition(const G4String& name, const G4ThreeVector& pos)
     71          { Position_vectorWrite(defineElement,"position",name,pos


gdml.py::

     * no handling of : firstposition, firstrotation


     166 class Boolean(Geometry):
     167     firstref = property(lambda self:self.elem.find("first").attrib["ref"])
     168     secondref = property(lambda self:self.elem.find("second").attrib["ref"])
     169 
     170     position = property(lambda self:self.find1_("position"))
     171     rotation = property(lambda self:self.find1_("rotation"))
     172     scale = None
     173     secondtransform = property(lambda self:construct_transform(self))
     174 
     175     first = property(lambda self:self.g.solids[self.firstref])
     176     second = property(lambda self:self.g.solids[self.secondref])
     177 
     ...
     183     def as_ncsg(self):
     ...
     188         left = self.first.as_ncsg()
     189         right = self.second.as_ncsg()
     ...
     194         right.transform = self.secondtransform
     195 
     196         cn = CSG(self.operation, name=self.name)
     197         cn.left = left
     198         cn.right = right
     199         return cn


::

      31 def construct_transform(obj):
      32     tla = obj.position.xyz if obj.position is not None else None
      33     rot = obj.rotation.xyz if obj.rotation is not None else None
      34     sca = obj.scale.xyz if obj.scale is not None else None
      35     order = "trs"
      36 
      37     #elem = filter(None, [tla,rot,sca])
      38     #if len(elem) > 1:
      39     #    log.warning("construct_transform multi %s " % repr(obj))
      40     #pass
      41 
      42     return make_transform( order, tla, rot, sca , three_axis_rotate=True, transpose_rotation=True, suppress_identity=False, dtype=np.float32 )
      43 


::

    258 def make_transform( order, tla, rot, sca, dtype=np.float32, suppress_identity=True, three_axis_rotate=False, transpose_rotation=False):
    259     """
    260     :param order: string containing "s" "r" and "t", standard order is "trs" meaning t*r*s  ie scale first, then rotate, then translate 
    261     :param tla: tx,ty,tz tranlation dists eg 0,0,0 for no translation 
    262     :param rot: ax,ay,az,angle_degrees  eg 0,0,1,45 for 45 degrees about z-axis
    263     :param sca: sx,sy,sz eg 1,1,1 for no scaling 
    264     :return mat: 4x4 numpy array 
    265 
    266     All arguments can be specified as comma delimited string, list or numpy array
    267 
    268     Translation of npy/tests/NGLMTest.cc:make_mat
    269     """
    270 
    271     if tla is None and rot is None and sca is None and suppress_identity:
    272         return None
    273 
    274     identity = np.eye(4, dtype=dtype)
    275     m = np.eye(4, dtype=dtype)
    276     for c in order:
    277         if c == 's':
    278             m = make_scale(sca, m)
    279         elif c == 'r':
    280             if three_axis_rotate:
    281                 m = rotate_three_axis(rot, m, transpose=transpose_rotation )
    282             else:
    283                 m = rotate(rot, m, transpose=transpose_rotation )
    284             pass
    285         elif c == 't':
    286             m = translate(tla, m)
    287         else:
    288             assert 0
    289         pass
    290     pass
    291 
    292     if suppress_identity and np.all( m == identity ):
    293         #log.warning("supressing identity transform")
    294         return None
    295     pass
    296     return m




FIXED : glTF viz shows messed up transforms
----------------------------------------------

Debug by editing the glTF to pick particular nodes::

    329578   "scenes": [
    329579     {
    329580       "nodes": [
    329581         3199
    329582       ]
    329583     }


::

   3199 : single pmt (with frame false looks correct, with frame true mangled)
   3155 : AD  (view starts from above the lid) (with frame false PMT all pointing in one direction, with frame true correct)
   3147 : pool with 2 ADs etc..


Similar trouble before
~~~~~~~~~~~~~~~~~~~~~~~~~

Every time, gets troubles from transforms...

* :doc:`gdml_gltf_transforms`


Debugging Approach ?
~~~~~~~~~~~~~~~~~~~~~~~

* compare the GGeo transforms from the two streams 
* simplify transform handling : avoid multiple holdings of transforms, 
  
Observations

* assembly of the PMT within its "frame" (of 5 parts) only involves 
  translation in z : so getting that correct could be deceptive as no rotation   


Switching to frame gets PMT pointing correct, but seems mangled inside themselves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* mangled : the base poking thru the front 


::

     20 glm::mat4* X4Transform3D::GetLocalTransform(const G4VPhysicalVolume* const pv, bool frame)
     21 {    
     22     glm::mat4* transform = NULL ;
     23     if(frame)
     24     {
     25         const G4RotationMatrix* rotp = pv->GetFrameRotation() ;
     26         G4ThreeVector    tla = pv->GetFrameTranslation() ;
     27         G4Transform3D    tra(rotp ? *rotp : G4RotationMatrix(),tla);
     28         transform = new glm::mat4(Convert( tra ));
     29     }   
     30     else
     31     {
     32         G4RotationMatrix rot = pv->GetObjectRotationValue() ;  // obj relative to mother
     33         G4ThreeVector    tla = pv->GetObjectTranslation() ; 
     34         G4Transform3D    tra(rot,tla);
     35         transform = new glm::mat4(Convert( tra ));
     36     }   
     37     return transform ;
     38 }   




FIXED : bad mesh association, missing meshes
------------------------------------------------

Also add metadata extras to allow to navigate the gltf.  Suspect 
are getting bad mesh association, as unexpected lots of repeated mesh.

Huh : only 35 meshes, (expect ~250) but the expected 12k nodes.

Suspect the lvIdx mesh identity.




