glfw_gleq_event_record_replay ?
====================================


Hmm could record/persist GLEQ events together with timestamps and then replay them in a 
somehow smoothed fashion : but that would be too low level to become a convenient source
format.  Actually that kind of think can be done separately from the application 
in an Applescript (hmm is there some Linux equivalent for GUI scripting ?).

Actually what would be more useful is a :doc:`scripted_view`
* https://www.gamasutra.com/blogs/MichaelKissner/20151027/257369/Writing_a_Game_Engine_from_Scratch__Part_1_Messaging.php


::

    epsilon:gleq blyth$ gleq-
    epsilon:gleq blyth$ gleq-cd
    epsilon:gleq blyth$ vi gleq.h 




::

    399 void Frame::listen()
    400 {
    401     glfwPollEvents();
    402 
    403     GLEQevent event;
    404     while (gleqNextEvent(&event))
    405     {
    406         if(m_dumpevent) dump_event(event);
    407         handle_event(event);
    408         gleqFreeEvent(&event);
    409     }
    410 }



::

    442 void OpticksViz::renderLoop()
    443 {
    ...
    458     while (!glfwWindowShouldClose(m_window))
    459     {
    460         m_frame->listen();
    ///       ^^^^^^^^^^^^^^^^^^^^^  GLFW is polled and acted upon here  
    461 
    462 #ifdef OPTICKS_NPYSERVER
    463         if(m_server) m_server->poll_one();
    464 #endif
    465         count = m_composition->tick();
    466 
    467         if(m_launcher)
    468         {
    469             m_launcher->launch(count);
    470         }
    471 
    472         if( m_composition->hasChanged() || m_interactor->hasChanged() || count == 1)
    473         {
    474             render();
    475             renderGUI();
    476 
    477             glfwSwapBuffers(m_window);
    478 
    479             m_interactor->setChanged(false);
    480             m_composition->setChanged(false);   // sets camera, view, trackball dirty status 
    481         }
    482     }
    483 }




