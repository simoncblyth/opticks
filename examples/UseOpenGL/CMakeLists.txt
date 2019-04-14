cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name UseOpenGL)
project(${name} VERSION 0.1.0)
include(OpticksBuildOptions)

find_package(OpenGL)   

message( STATUS "OPENGL_FOUND          : ${OPENGL_FOUND}")
message( STATUS "OpenGL_GL_PREFERENCE  : ${OpenGL_GL_PREFERENCE} ")
message( STATUS "OPENGL_LIBRARIES      : ${OPENGL_LIBRARIES}")
message( STATUS "OPENGL_egl_LIBRARY    : ${OPENGL_egl_LIBRARY}")
message( STATUS "OPENGL_glu_LIBRARY    : ${OPENGL_glu_LIBRARY}")
message( STATUS "OPENGL_glx_LIBRARY    : ${OPENGL_glx_LIBRARY}      (GLVND 'GLX' Library)")
message( STATUS "OPENGL_opengl_LIBRARY : ${OPENGL_opengl_LIBRARY}   (GLVND 'OpenGL' Library)")
message( STATUS "OPENGL_gl_LIBRARY     : ${OPENGL_gl_LIBRARY}       (LEGACY 'OpenGL' Library)")


echo_pfx_vars(OpenGL "FOUND;DIR;VERSION;VERSION_INTEGER;INCLUDE_DIR;INCLUDE_DIRS;LIBRARIES;DEFINITIONS")


#[=[

https://cmake.org/cmake/help/latest/module/FindOpenGL.html

OpenGL_GL_PREFERENCE


-- Configuring UseOpenGL
CMake Warning (dev) at /usr/share/cmake3/Modules/FindOpenGL.cmake:270 (message):
  Policy CMP0072 is not set: FindOpenGL prefers GLVND by default when
  available.  Run "cmake --help-policy CMP0072" for policy details.  Use the
  cmake_policy command to set the policy and suppress this warning.

  FindOpenGL found both a legacy GL library:

    OPENGL_gl_LIBRARY: /usr/lib64/libGL.so

  and GLVND libraries for OpenGL and GLX:

    OPENGL_opengl_LIBRARY: /usr/lib64/libOpenGL.so
    OPENGL_glx_LIBRARY: /usr/lib64/libGLX.so

  OpenGL_GL_PREFERENCE has not been set to "GLVND" or "LEGACY", so for
  compatibility with CMake 3.10 and below the legacy GL library will be used.
Call Stack (most recent call first):
  CMakeLists.txt:6 (find_package)
This warning is for project developers.  Use -Wno-dev to suppress it.

-- Found OpenGL: /usr/lib64/libOpenGL.so   
 key='OpenGL_FOUND' val='TRUE' 
 key='OpenGL_DIR' val='' 
 key='OpenGL_VERSION' val='' 
 key='OpenGL_VERSION_INTEGER' val='' 
 key='OpenGL_INCLUDE_DIR' val='' 
 key='OpenGL_INCLUDE_DIRS' val='' 
 key='OpenGL_LIBRARIES' val='' 
 key='OpenGL_DEFINITIONS' val='' 





[blyth@localhost lib64]$ ldd /home/blyth/local/opticks/externals/lib64/libGLEW.so.1.13
ldd: warning: you do not have execution permission for `/home/blyth/local/opticks/externals/lib64/libGLEW.so.1.13'
    linux-vdso.so.1 =>  (0x00007ffd9e2e0000)
    libGL.so.1 (0x00007fd4c36f6000)
    libX11.so.6 (0x00007fd4c33b8000)
    libc.so.6 (0x00007fd4c2feb000)
    libGLX.so.0 (0x00007fd4c2db9000)
    libXext.so.6 (0x00007fd4c2ba7000)
    libGLdispatch.so.0 (0x00007fd4c28f1000)
    libdl.so.2 (0x00007fd4c26ed000)
    libpthread.so.0 (0x00007fd4c24d1000)
    libxcb.so.1 (0x00007fd4c22a9000)
    /lib64/ld-linux-x86-64.so.2 (0x00007fd4c3c1d000)
    libXau.so.6 (0x00007fd4c20a5000)
[blyth@localhost lib64]$ 




#]=]




