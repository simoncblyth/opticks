glfw-update-perhaps
=====================

The CMake based installation of GLFW installs CONFIG-wise
but cmake/Modules/FindOpticksGLFW.cmake developed agains an old GLFW 3.1 
is ignoring it.  Potentially the new 3.3 install is less deficient::

    -- Install configuration: ""
    -- Up-to-date: /usr/local/opticks/externals/include/GLFW
    -- Installing: /usr/local/opticks/externals/include/GLFW/glfw3.h
    -- Installing: /usr/local/opticks/externals/include/GLFW/glfw3native.h
    -- Installing: /usr/local/opticks/externals/lib/cmake/glfw3/glfw3Config.cmake
    -- Installing: /usr/local/opticks/externals/lib/cmake/glfw3/glfw3ConfigVersion.cmake
    -- Installing: /usr/local/opticks/externals/lib/cmake/glfw3/glfw3Targets.cmake
    -- Installing: /usr/local/opticks/externals/lib/cmake/glfw3/glfw3Targets-noconfig.cmake
    -- Installing: /usr/local/opticks/externals/lib/pkgconfig/glfw3.pc
    -- Installing: /usr/local/opticks/externals/lib/libglfw.3.3.dylib
    -- Installing: /usr/local/opticks/externals/lib/libglfw.3.dylib
    -- Up-to-date: /usr/local/opticks/externals/lib/libglfw.dylib


