GLFW/OpenGL GPU Hang
=======================

Ray trace is currently performing better than the rasterized... 
need to use less tris in the polygonization.


::

    tgltf-;tgltf-gdml


    Scene::nextGeometryStyle : bbox 
    GPU hang occurred, msgtracer returned -1
    Process 23299 stopped
    * thread #1: tid = 0x2f9d18, 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8f018866:  jae    0x7fff8f018870            ; __pthread_kill + 20
       0x7fff8f018868:  movq   %rax, %rdi
       0x7fff8f01886b:  jmp    0x7fff8f015175            ; cerror_nocancel
       0x7fff8f018870:  retq   
    (lldb) 



    Scene::nextGeometryStyle : none 
    Scene::nextGeometryStyle : wire 
    Scene::nextGeometryStyle : bbox 
    GPU hang occurred, msgtracer returned -1
    Process 24028 stopped
    * thread #1: tid = 0x2fb7db, 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8f018866:  jae    0x7fff8f018870            ; __pthread_kill + 20
       0x7fff8f018868:  movq   %rax, %rdi
       0x7fff8f01886b:  jmp    0x7fff8f015175            ; cerror_nocancel
       0x7fff8f018870:  retq   
    (lldb) bt
    * thread #1: tid = 0x2fb7db, 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff866b535c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8d405b1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff880c61ca libGPUSupportMercury.dylib`gpusKillClient + 111
        frame #4: 0x00007fff880c751c libGPUSupportMercury.dylib`gpusSubmitDataBuffers + 161
        frame #5: 0x00001234402f51c3 GeForceGLDriver`___lldb_unnamed_function10892$$GeForceGLDriver + 360
        frame #6: 0x00001234402f55cd GeForceGLDriver`gldPresentFramebufferData + 136
        frame #7: 0x00007fff8ce640ed GLEngine`glSwap_Exec + 93
        frame #8: 0x00007fff886ed089 OpenGL`CGLFlushDrawable + 66
        frame #9: 0x00000001020dade3 libglfw.3.dylib`_glfwPlatformSwapBuffers + 35
        frame #10: 0x00000001020cdb48 libglfw.3.dylib`glfwSwapBuffers + 72
        frame #11: 0x00000001021efc16 libOGLRap.dylib`OpticksViz::renderLoop(this=0x0000000109486710) + 934 at OpticksViz.cc:453
        frame #12: 0x00000001021ef332 libOGLRap.dylib`OpticksViz::visualize(this=0x0000000109486710) + 34 at OpticksViz.cc:130
        frame #13: 0x000000010396e92f libOK.dylib`OKMgr::visualize(this=0x00007fff5fbfe8f8) + 47 at OKMgr.cc:113
        frame #14: 0x000000010000a95a OKTest`main(argc=23, argv=0x00007fff5fbfe9d0) + 1402 at OKTest.cc:62
        frame #15: 0x00007fff8a48b5fd libdyld.dylib`start + 1
    (lldb) 


    (lldb) f 11
    frame #11: 0x00000001021efc16 libOGLRap.dylib`OpticksViz::renderLoop(this=0x0000000109486710) + 934 at OpticksViz.cc:453
       450              render();
       451              renderGUI();
       452  
    -> 453              glfwSwapBuffers(m_window);
       454  
       455              m_interactor->setChanged(false);  
       456              m_composition->setChanged(false);   // sets camera, view, trackball dirty status 
    (lldb) 


