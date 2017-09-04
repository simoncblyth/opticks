j1707 near/far tweaks
=========================

Initial oglrap view is confusing due to bad near/far.



Workaround is to target a smaller volume and ajust eye point

::

    --target 12 --eye 0.5,0.5,0.0


    op --j1707 --gltf 3 --tracer  --instcull --lod 1 --lodconfig "levels=3,verbosity=2,instanced_lodify_onload=1" --debugger --target 12 --eye 0.5,0.5,0.0
    op --j1707 --gltf 3 --tracer  --instcull --lod 1 --lodconfig "levels=3,verbosity=2,instanced_lodify_onload=1" --debugger --target 12 --eye 0.5,0.5,0.0
    op --j1707 --gltf 3 --tracer  --instcull --lod 1 --debugger 
    op --j1707 --gltf 3 --tracer  --instcull --lod 1 --debugger --target 12 --eye 0.5,0.5,0.0



::

   550     <box lunit="mm" name="sWorld0x14d9850" x="120000" y="120000" z="120000"/>
   551   </solids>
   552 
    

::

    2017-09-04 13:08:49.503 INFO  [2976304] [OpticksViz::uploadGeometry@264] OpticksViz::uploadGeometry setting target 0
    2017-09-04 13:08:49.503 FATAL [2976304] [OpticksGeometry::setTarget@131] OpticksGeometry::setTarget  based on CenterExtent from m_mesh0  target 0 aim 1 ce  0 0 0 60000
    2017-09-04 13:08:49.503 INFO  [2976304] [Composition::setCenterExtent@992] Composition::setCenterExtent ce 0.0000,0.0000,0.0000,60000.0000



::

    delta:opticksgeo blyth$ opticks-find setTarget 
    ./ggeo/GColorizer.cc:void GColorizer::setTarget(nvec3* target)
    ./ggeo/GColorizer.cc:    setTarget( reinterpret_cast<nvec3*>(vertex_colors) );
    ./ggeo/GColorizer.cc:        LOG(warning) << "GColorizer::traverse must setTarget before traverse " ;
    ./oglrap/AxisApp.cc:    m_scene->setTarget(0, true);
    ./oglrap/OpticksViz.cc:    m_geometry->setTarget(target, autocam);
    ./oglrap/Scene.cc:    if(     strcmp(name,TARGET)==0)    setTarget(v);
    ./oglrap/Scene.cc:        setTarget(value);   
    ./oglrap/Scene.cc:        m_hub->setTarget(m_touch);
    ./oglrap/Scene.cc:void Scene::setTarget(unsigned int target, bool aim)
    ./oglrap/Scene.cc:    m_hub->setTarget(target, aim); // sets center_extent in Composition via okg-/OpticksHub/OpticksGeometry
    ./oglrap/tests/SceneCheck.cc:    m_scene->setTarget(target, autocam);
    ./optickscore/Animator.cc:void Animator::setTarget(float* target)
    ./optickscore/Animator.cc:    setTargetValue(val);
    ./optickscore/Animator.cc:void Animator::setTargetValue(float val)
    ./optickscore/Composition.cc:    // this is invoked by App::uploadGeometry/Scene::setTarget

    ./opticksgeo/OpticksGeometry.cc:void  OpticksGeometry::setTarget(unsigned target, bool aim)
    ./opticksgeo/OpticksGeometry.cc:        LOG(info) << "OpticksGeometry::setTarget " << target << " deferring as geometry not loaded " ; 
    ./opticksgeo/OpticksGeometry.cc:    LOG(fatal)<<"OpticksGeometry::setTarget " 

    ./opticksgeo/OpticksHub.cc:void OpticksHub::setTarget(unsigned target, bool aim)
    ./opticksgeo/OpticksHub.cc:    m_geometry->setTarget(target, aim );

    ./ggeo/GColorizer.hh:        void setTarget(nvec3* target);  // where to write the colors
    ./oglrap/Scene.hh:      //  void setTarget(unsigned int index=0, bool aim=true); 
    ./oglrap/Scene.hh:        void setTarget(unsigned int index=0, bool aim=true); 
    ./optickscore/Animator.hh:        void          setTargetValue(float value);
    ./optickscore/Animator.hh:        void          setTarget(float* target); // qty to be stepped
    ./opticksgeo/OpticksGeometry.hh:       void            setTarget(unsigned target=0, bool aim=true);
    ./opticksgeo/OpticksHub.hh:       void setTarget(unsigned target=0, bool aim=true);
    ./opticksnpy/TorchStepNPY.cpp:        case TARGET         : setTargetLocal(s)    ;break;
    ./opticksnpy/TorchStepNPY.cpp:void TorchStepNPY::setTargetLocal(const char* s)
    ./opticksnpy/TorchStepNPY.hpp:       void setTargetLocal(const char* s );
    delta:opticks blyth$ 
    delta:opticks blyth$ 


::

    (lldb) bt
    * thread #1: tid = 0x2d97b1, 0x00007fff95594866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff95594866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff8cc3135c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff93981b1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff9394b9bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x0000000102284d7e libOpticksGeometry.dylib`OpticksGeometry::setTarget(this=0x0000000105b26440, target=0, aim=true) + 62 at OpticksGeometry.cc:119
        frame #5: 0x000000010248466e libOGLRap.dylib`OpticksViz::uploadGeometry(this=0x0000000113cb4ee0) + 782 at OpticksViz.cc:266
        frame #6: 0x0000000102484072 libOGLRap.dylib`OpticksViz::init(this=0x0000000113cb4ee0) + 802 at OpticksViz.cc:124
        frame #7: 0x0000000102483d0a libOGLRap.dylib`OpticksViz::OpticksViz(this=0x0000000113cb4ee0, hub=0x0000000105b22490, idx=0x0000000113cb27a0, immediate=true) + 362 at OpticksViz.cc:86
        frame #8: 0x0000000102484194 libOGLRap.dylib`OpticksViz::OpticksViz(this=0x0000000113cb4ee0, hub=0x0000000105b22490, idx=0x0000000113cb27a0, immediate=true) + 52 at OpticksViz.cc:88
        frame #9: 0x0000000103c11300 libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfe9e8, argc=13, argv=0x00007fff5fbfeac0, argforced=0x000000010001580d) + 544 at OKMgr.cc:43
        frame #10: 0x0000000103c1164b libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfe9e8, argc=13, argv=0x00007fff5fbfeac0, argforced=0x000000010001580d) + 43 at OKMgr.cc:49
        frame #11: 0x000000010000a95d OTracerTest`main(argc=13, argv=0x00007fff5fbfeac0) + 1373 at OTracerTest.cc:64
        frame #12: 0x00007fff90a075fd libdyld.dylib`start + 1
    (lldb) 


