oglrap-compilation-fail-junoenv-missing-new-opengl-symbols
==============================================================

Report from Artem, missing GL_GEOMETRY_SHADER GL_CONTEXT_LOST
-----------------------------------------------------------------

I am trying to install opticks support in juno soft and found...::

    [opticks-make] /cern/juno/ExternalLibs/Build/opticks-0.0.0-rc3/oglrap/G.cc: In static member function 'static const char* G::Shader(GLenum)':
    [opticks-make] /cern/juno/ExternalLibs/Build/opticks-0.0.0-rc3/oglrap/G.cc:50:13: error: 'GL_GEOMETRY_SHADER' was not declared in this scope; did you mean 'GL_GEOMETRY_SHADER_'?
    [opticks-make]    50 |        case GL_GEOMETRY_SHADER: s = GL_GEOMETRY_SHADER_ ; break ;
    [opticks-make]       |             ^~~~~~~~~~~~~~~~~~
    [opticks-make]       |             GL_GEOMETRY_SHADER_
    [opticks-make] /cern/juno/ExternalLibs/Build/opticks-0.0.0-rc3/oglrap/G.cc: In static member function 'static const char* G::Err(GLenum)':
    [opticks-make] /cern/juno/ExternalLibs/Build/opticks-0.0.0-rc3/oglrap/G.cc:68:14: error: 'GL_CONTEXT_LOST' was not declared in this scope; did you mean 'GL_CONTEXT_LOST_'?
    [opticks-make]    68 |         case GL_CONTEXT_LOST : s = GL_CONTEXT_LOST_ ; break ;


Probable Cause : build finding ancient glew.h
-----------------------------------------------

The build is probably finding an ancient version of glew.h which does not 
have the necessary OpenGL symbols. 


/Users/blyth/junotop/ExternalLibs/Opticks/0.0.0-rc1/externals/glew/glew-1.13.0/include/GL/glew.h


Relevant Opticks Examples
--------------------------


examples/UseOpticksGLEW
    CMake finding GLEW and version dumping 

examples/UseOpenGL
    CMake finding OpenGL (not so useful as all use of OpenGL is via glew or glfw)

examples/UseGeometryShader
    near minimal example using OpenGL/GLEW to setup shaders and popup a window



libGLEW.so glew.h Issue
-------------------------

OGLRap compilation fails when build against JUNO externals, due to 
the GL/glew.h coming from ROOT lacking defines from modern OpenGL.

Investigating this below, observe that the glew.h from ROOT stops 
at OpenGL 3.0

See examples/UseOpticksGLEW


Dirty Fix
------------

Fixed after manual edit of JUNOTOP/bashrc.sh excluding ROOT which brings in wrong libGLEW.so::

     13 source /home/blyth/junotop/ExternalLibs/CLHEP/2.4.1.0/bashrc
     14 source /home/blyth/junotop/ExternalLibs/xrootd/4.10.0/bashrc
     15 #source /home/blyth/junotop/ExternalLibs/ROOT/6.18.00/bashrc
     16 source /home/blyth/junotop/ExternalLibs/HepMC/2.06.09/bashrc
     17 source /home/blyth/junotop/ExternalLibs/Geant4/10.05.p01/bashrc


Attempt for a better fix by rebuilding ROOT 
---------------------------------------------

Remove the old lib and headers::

    rm -rf /home/blyth/junotop/ExternalLibs/ROOT/6.18.00/include/GL
    [blyth@localhost junoenv]$ rm /home/blyth/junotop/ExternalLibs/ROOT/6.18.00/lib/libGLEW.so

Then try to rebuild ROOT with "-Dbuiltin_glew=OFF"::

    [blyth@localhost ~]$ cd $JUNOTOP/junoenv
    [blyth@localhost junoenv]$ bash junoenv libs all ROOT


junoenv switch off the ancient glew that comes with ROOT::

    174 function juno-ext-libs-ROOT-conf-cmake {
    175     local msg="===== $FUNCNAME: "
    176     cmake .. -DCMAKE_INSTALL_PREFIX=$(juno-ext-libs-ROOT-install-dir) \
    177           -DVc=ON \
    178           -DVecCore=ON \
    179           -Dxrootd=ON \
    180           -Dminuit2=ON \
    181           -Droofit=ON \
    182           -Dtbb=ON \
    183           -Dgdml=ON \
    184           -Dcastor=OFF \
    185           -Drfio=OFF \
    186           -Dsqlite=ON \
    187           -DGSL_DIR=$(juno-ext-libs-gsl-install-dir) \
    188           -DFFTW3_DIR=$(juno-ext-libs-fftw3-install-dir) \
    189           -DTBB=$(juno-ext-libs-tbb-install-dir) \
    190           -DXROOTD_ROOT_DIR=$(juno-ext-libs-xrootd-install-dir) \
    191           -DXROOTD_INCLUDE_DIR=$(juno-ext-libs-xrootd-install-dir)/include/xrootd \
    192           -DSQLITE_LIBRARIES=$(juno-ext-libs-sqlite3-install-dir)/lib/libsqlite3.so \
    193           -Dbuiltin_glew=OFF
    194     # SCB: dont use builtin GLEW as it is ancient and breaks Opticks build against JUNO externals
    195 }

That is in SVN, however does mean users will have to rebuild ROOT::

    epsilon:packages blyth$ pwd
    /Users/blyth/junotop/junoenv/packages

    epsilon:packages blyth$ svn log ROOT.sh -l3
    ------------------------------------------------------------------------
    r3912 | blyth | 2020-05-21 14:29:37 +0100 (Thu, 21 May 2020) | 1 line

    Opticks integration requires ROOT to not use an ancient builtin GLEW, and Geant4 to not use builtin G4clhep 
    ------------------------------------------------------------------------
    r3826 | lintao | 2020-03-26 13:20:03 +0000 (Thu, 26 Mar 2020) | 1 line

    WIP: upgrade ROOT from 6.18.00 to 6.20.02
    ------------------------------------------------------------------------
    r3554 | lintao | 2019-08-19 16:38:32 +0100 (Mon, 19 Aug 2019) | 1 line

    update ROOT and add sqlite3.
    ------------------------------------------------------------------------
    epsilon:packages blyth$ 


Hmm, this is not what we want : ROOT build should be independent of Opticks one::

    [ROOT-conf] -- Could NOT find TIFF (missing: TIFF_LIBRARY TIFF_INCLUDE_DIR) 
    [ROOT-conf] -- Building AfterImage library included in ROOT itself
    [ROOT-conf] -- Looking for GSL
    [ROOT-conf] -- Looking for python
    [ROOT-conf] -- Looking for OpenGL
    [ROOT-conf] -- Looking for GLEW
    [ROOT-conf] -- Found GLEW: /home/blyth/local/opticks/externals/include (found version "1.13.0") 
    [ROOT-conf] -- Looking for LibXml2
    [ROOT-conf] -- Looking for MySQL
    [ROOT-conf] -- Looking for SQLite
    [ROOT-conf] -- Looking for FFTW3
    [ROOT-conf] -- Looking for CFITSIO
    [ROOT-conf] -- Looking for XROOTD

After preventing running om- in environment by default which causes om-export to set CMAKE_PREFIX_PATH::

    [ROOT-conf] -- Could NOT find TIFF (missing: TIFF_LIBRARY TIFF_INCLUDE_DIR) 
    [ROOT-conf] -- Building AfterImage library included in ROOT itself
    [ROOT-conf] -- Looking for GSL
    [ROOT-conf] -- Looking for python
    [ROOT-conf] -- Looking for OpenGL
    [ROOT-conf] -- Looking for GLEW
    [ROOT-conf] -- Looking for LibXml2
    [ROOT-conf] -- Looking for MySQL
    [ROOT-conf] -- Looking for SQLite
    [ROOT-conf] -- Looking for FFTW3
    [ROOT-conf] -- Looking for CFITSIO
    [ROOT-conf] -- Looking for XROOTD
    [ROOT-conf] -- Looking for TBB
    [ROOT-conf] -- Looking for BLAS for optional parts of TMVA



Investigating the issue : compare the glew.h
------------------------------------------------

* Geometry shaders were introduces at about OpenGL 3.2 

::

    [blyth@localhost junotop]$ l $JUNOTOP/ExternalLibs/ROOT/6.18.00/include/GL/
    total 668
    -rw-r--r--. 1 blyth blyth 624322 Jun 25  2019 glew.h
    -rw-r--r--. 1 blyth blyth  55303 Jun 25  2019 glxew.h

    epsilon:glew-1.13.0 blyth$ l /usr/local/opticks/externals/glew/glew-1.13.0/include/GL/
    total 2312
    -rw-rw-r--  1 blyth  staff  1038562 Aug 10  2015 glew.h
    -rw-rw-r--  1 blyth  staff    74912 Aug 10  2015 glxew.h
    -rw-rw-r--  1 blyth  staff    64836 Aug 10  2015 wglew.h


/usr/local/opticks/externals/glew/glew-1.13.0/include/GL/glew.h::

     2333 /* ----------------------------- GL_VERSION_3_2 ---------------------------- */
     2334 
     2335 #ifndef GL_VERSION_3_2
     2336 #define GL_VERSION_3_2 1
     2337 
     2338 #define GL_CONTEXT_CORE_PROFILE_BIT 0x00000001
     2339 #define GL_CONTEXT_COMPATIBILITY_PROFILE_BIT 0x00000002
     2340 #define GL_LINES_ADJACENCY 0x000A
     2341 #define GL_LINE_STRIP_ADJACENCY 0x000B
     2342 #define GL_TRIANGLES_ADJACENCY 0x000C
     2343 #define GL_TRIANGLE_STRIP_ADJACENCY 0x000D
     2344 #define GL_PROGRAM_POINT_SIZE 0x8642
     2345 #define GL_GEOMETRY_VERTICES_OUT 0x8916
     2346 #define GL_GEOMETRY_INPUT_TYPE 0x8917
     2347 #define GL_GEOMETRY_OUTPUT_TYPE 0x8918
     2348 #define GL_MAX_GEOMETRY_TEXTURE_IMAGE_UNITS 0x8C29
     2349 #define GL_FRAMEBUFFER_ATTACHMENT_LAYERED 0x8DA7
     2350 #define GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS 0x8DA8
     2351 #define GL_GEOMETRY_SHADER 0x8DD9
     2352 #define GL_MAX_GEOMETRY_UNIFORM_COMPONENTS 0x8DDF
     2353 #define GL_MAX_GEOMETRY_OUTPUT_VERTICES 0x8DE0
     2354 #define GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS 0x8DE1

::

    epsilon:glew-1.13.0 blyth$ grep define\ GL_VERSION_ /usr/local/opticks/externals/glew/glew-1.13.0/include/GL/glew.h
    #define GL_VERSION_1_1 1
    #define GL_VERSION_1_2 1
    #define GL_VERSION_1_2_1 1
    #define GL_VERSION_1_3 1
    #define GL_VERSION_1_4 1
    #define GL_VERSION_1_5 1
    #define GL_VERSION_2_0 1
    #define GL_VERSION_2_1 1
    #define GL_VERSION_3_0 1
    #define GL_VERSION_3_1 1
    #define GL_VERSION_3_2 1
    #define GL_VERSION_3_3 1
    #define GL_VERSION_4_0 1
    #define GL_VERSION_4_1 1
    #define GL_VERSION_4_2 1
    #define GL_VERSION_4_3 1
    #define GL_VERSION_4_4 1
    #define GL_VERSION_4_5 1
    epsilon:glew-1.13.0 blyth$ 

    epsilon:glew-1.13.0 blyth$ grep define\ GLEW_VERSION /usr/local/opticks/externals/glew/glew-1.13.0/include/GL/glew.h
    #define GLEW_VERSION_1_1 GLEW_GET_VAR(__GLEW_VERSION_1_1)
    #define GLEW_VERSION_1_2 GLEW_GET_VAR(__GLEW_VERSION_1_2)
    #define GLEW_VERSION_1_2_1 GLEW_GET_VAR(__GLEW_VERSION_1_2_1)
    #define GLEW_VERSION_1_3 GLEW_GET_VAR(__GLEW_VERSION_1_3)
    #define GLEW_VERSION_1_4 GLEW_GET_VAR(__GLEW_VERSION_1_4)
    #define GLEW_VERSION_1_5 GLEW_GET_VAR(__GLEW_VERSION_1_5)
    #define GLEW_VERSION_2_0 GLEW_GET_VAR(__GLEW_VERSION_2_0)
    #define GLEW_VERSION_2_1 GLEW_GET_VAR(__GLEW_VERSION_2_1)
    #define GLEW_VERSION_3_0 GLEW_GET_VAR(__GLEW_VERSION_3_0)
    #define GLEW_VERSION_3_1 GLEW_GET_VAR(__GLEW_VERSION_3_1)
    #define GLEW_VERSION_3_2 GLEW_GET_VAR(__GLEW_VERSION_3_2)
    #define GLEW_VERSION_3_3 GLEW_GET_VAR(__GLEW_VERSION_3_3)
    #define GLEW_VERSION_4_0 GLEW_GET_VAR(__GLEW_VERSION_4_0)
    #define GLEW_VERSION_4_1 GLEW_GET_VAR(__GLEW_VERSION_4_1)
    #define GLEW_VERSION_4_2 GLEW_GET_VAR(__GLEW_VERSION_4_2)
    #define GLEW_VERSION_4_3 GLEW_GET_VAR(__GLEW_VERSION_4_3)
    #define GLEW_VERSION_4_4 GLEW_GET_VAR(__GLEW_VERSION_4_4)
    #define GLEW_VERSION_4_5 GLEW_GET_VAR(__GLEW_VERSION_4_5)
    #define GLEW_VERSION 1
    #define GLEW_VERSION_MAJOR 2
    #define GLEW_VERSION_MINOR 3
    #define GLEW_VERSION_MICRO 4
    epsilon:glew-1.13.0 blyth$ 


The glew.h from ROOT just doesnt have the symbols::

    [blyth@localhost ~]$ grep define\ GL_VERSION_ /home/blyth/junotop/ExternalLibs/ROOT/6.18.00/include/GL/glew.h
    #define GL_VERSION_1_1 1
    #define GL_VERSION_1_2 1
    #define GL_VERSION_1_3 1
    #define GL_VERSION_1_4 1
    #define GL_VERSION_1_5 1
    #define GL_VERSION_2_0 1
    #define GL_VERSION_2_1 1
    #define GL_VERSION_3_0 1
    [blyth@localhost ~]$ 

    [blyth@localhost GL]$ grep define\ GL_VERSION_ /home/blyth/junotop/ExternalLibs/Build/root-6.18.00/root-6.18.00/graf3d/glew/inc/GL/glew.h
    #define GL_VERSION_1_1 1
    #define GL_VERSION_1_2 1
    #define GL_VERSION_1_3 1
    #define GL_VERSION_1_4 1
    #define GL_VERSION_1_5 1
    #define GL_VERSION_2_0 1
    #define GL_VERSION_2_1 1
    #define GL_VERSION_3_0 1


    [blyth@localhost ~]$ grep define\ GLEW_VERSION /home/blyth/junotop/ExternalLibs/ROOT/6.18.00/include/GL/glew.h
    #define GLEW_VERSION_1_1 GLEW_GET_VAR(__GLEW_VERSION_1_1)
    #define GLEW_VERSION_1_2 GLEW_GET_VAR(__GLEW_VERSION_1_2)
    #define GLEW_VERSION_1_3 GLEW_GET_VAR(__GLEW_VERSION_1_3)
    #define GLEW_VERSION_1_4 GLEW_GET_VAR(__GLEW_VERSION_1_4)
    #define GLEW_VERSION_1_5 GLEW_GET_VAR(__GLEW_VERSION_1_5)
    #define GLEW_VERSION_2_0 GLEW_GET_VAR(__GLEW_VERSION_2_0)
    #define GLEW_VERSION_2_1 GLEW_GET_VAR(__GLEW_VERSION_2_1)
    #define GLEW_VERSION_3_0 GLEW_GET_VAR(__GLEW_VERSION_3_0)
    #define GLEW_VERSION 1
    #define GLEW_VERSION_MAJOR 2
    #define GLEW_VERSION_MINOR 3
    #define GLEW_VERSION_MICRO 4
    [blyth@localhost ~]$ 


Investigate the ROOT build
------------------------------


/home/blyth/junotop/ExternalLibs/Build/root-6.18.00/root-6.18.00/graf3d/CMakeLists.txt::

     16 if (opengl)
     17    add_subdirectory(eve) # special CMakeLists.txt
     18    add_subdirectory(gl) # special CMakeLists.txt
     19    if(builtin_glew)
     20       add_subdirectory(glew)
     21    endif()
     22    if(builtin_ftgl)
     23       add_subdirectory(ftgl)
     24    endif()
     25   add_subdirectory(gviz3d) # special CMakeLists.txt
     26 endif()

::

    [blyth@localhost modules]$ grep builtin *.cmake
    RootBuildOptions.cmake:ROOT_BUILD_OPTION(builtin_afterimage ON "Build bundled copy of libAfterImage")
    RootBuildOptions.cmake:ROOT_BUILD_OPTION(builtin_cfitsio OFF "Build CFITSIO internally (requires network)")
    RootBuildOptions.cmake:ROOT_BUILD_OPTION(builtin_clang ON "Build bundled copy of Clang")
    RootBuildOptions.cmake:ROOT_BUILD_OPTION(builtin_davix OFF "Build Davix internally (requires network)")
    RootBuildOptions.cmake:ROOT_BUILD_OPTION(builtin_fftw3 OFF "Build FFTW3 internally (requires network)")
    RootBuildOptions.cmake:ROOT_BUILD_OPTION(builtin_freetype OFF "Build bundled copy of freetype")
    RootBuildOptions.cmake:ROOT_BUILD_OPTION(builtin_ftgl ON "Build bundled copy of FTGL")
    RootBuildOptions.cmake:ROOT_BUILD_OPTION(builtin_gl2ps OFF "Build bundled copy of gl2ps")
    RootBuildOptions.cmake:ROOT_BUILD_OPTION(builtin_glew ON "Build bundled copy of GLEW")
    RootBuildOptions.cmake:ROOT_BUILD_OPTION(builtin_gsl OFF "Build GSL internally (requires network)")
    RootBuildOptions.cmake:ROOT_BUILD_OPTION(builtin_llvm ON "Build bundled copy of LLVM")
    RootBuildOptions.cmake:ROOT_BUILD_OPTION(builtin_lz4 OFF "Build bundled copy of lz4")

* https://root.cern.ch/building-root#options

::

    [blyth@localhost glew]$ pwd
    /home/blyth/junotop/ExternalLibs/Build/root-6.18.00/root-6.18.00/graf3d/glew
    [blyth@localhost glew]$ find . 
    .
    ./CMakeLists.txt
    ./inc
    ./inc/GL
    ./inc/GL/glew.h
    ./inc/GL/glxew.h
    ./inc/GL/wglew.h
    ./isystem
    ./isystem/GL
    ./isystem/GL/gl.h
    ./isystem/OpenGL
    ./isystem/OpenGL/gl.h
    ./src
    ./src/glew.c
    [blyth@localhost glew]$ 



Check on root forum
---------------------

* https://root-forum.cern.ch/t/what-is-mt-option-when-compiling-root-6-12-and-compilation-failure-within-builtin-glew/28806/4

I think there is a typo in our build system that fails to detect the case when
OpenGL is found, but not GLU. I will look into it and fix, but for your build
you can simply do sudo apt-get install libglew-dev and disable builtin_glew in
ROOT.


JUNO root build config
-----------------------

::

    173 function juno-ext-libs-ROOT-conf-cmake {
    174     local msg="===== $FUNCNAME: "
    175     cmake .. -DCMAKE_INSTALL_PREFIX=$(juno-ext-libs-ROOT-install-dir) \
    176           -DVc=ON \
    177           -DVecCore=ON \
    178           -Dxrootd=ON \
    179           -Dminuit2=ON \
    180           -Droofit=ON \
    181           -Dtbb=ON \
    182           -Dgdml=ON \
    183           -Dcastor=OFF \
    184           -Drfio=OFF \
    185           -Dsqlite=ON \
    186           -DGSL_DIR=$(juno-ext-libs-gsl-install-dir) \
    187           -DFFTW3_DIR=$(juno-ext-libs-fftw3-install-dir) \
    188           -DTBB=$(juno-ext-libs-tbb-install-dir) \
    189           -DXROOTD_ROOT_DIR=$(juno-ext-libs-xrootd-install-dir) \
    190           -DXROOTD_INCLUDE_DIR=$(juno-ext-libs-xrootd-install-dir)/include/xrootd \
    191           -DSQLITE_LIBRARIES=$(juno-ext-libs-sqlite3-install-dir)/lib/libsqlite3.so
    192 }



Check the system GLEW on Linux and macports one on Darwin
------------------------------------------------------------


::

    [blyth@localhost ~]$ repoquery --list glew-devel.x86_64 | grep glew.h
    /usr/include/GL/glew.h
    /usr/include/GL/wglew.h
    /usr/share/doc/glew-devel-1.10.0/glew.html
    /usr/share/doc/glew-devel-1.10.0/wglew.html

::

    [blyth@localhost ~]$ grep define\ GL_VERSION /usr/include/GL/glew.h
    #define GL_VERSION_1_1 1
    #define GL_VERSION 0x1F02
    #define GL_VERSION_1_2 1
    #define GL_VERSION_1_2_1 1
    #define GL_VERSION_1_3 1
    #define GL_VERSION_1_4 1
    #define GL_VERSION_1_5 1
    #define GL_VERSION_2_0 1
    #define GL_VERSION_2_1 1
    #define GL_VERSION_3_0 1
    #define GL_VERSION_3_1 1
    #define GL_VERSION_3_2 1
    #define GL_VERSION_3_3 1
    #define GL_VERSION_4_0 1
    #define GL_VERSION_4_1 1
    #define GL_VERSION_4_2 1
    #define GL_VERSION_4_3 1
    #define GL_VERSION_4_4 1


    epsilon:glew-1.13.0 blyth$ grep define\ GL_VERSION_ /opt/local/include/GL/glew.h
    #define GL_VERSION_1_1 1
    #define GL_VERSION_1_2 1
    #define GL_VERSION_1_2_1 1
    #define GL_VERSION_1_3 1
    #define GL_VERSION_1_4 1
    #define GL_VERSION_1_5 1
    #define GL_VERSION_2_0 1
    #define GL_VERSION_2_1 1
    #define GL_VERSION_3_0 1
    #define GL_VERSION_3_1 1
    #define GL_VERSION_3_2 1
    #define GL_VERSION_3_3 1
    #define GL_VERSION_4_0 1
    #define GL_VERSION_4_1 1
    #define GL_VERSION_4_2 1
    #define GL_VERSION_4_3 1
    #define GL_VERSION_4_4 1
    #define GL_VERSION_4_5 1
    #define GL_VERSION_4_6 1










All newish OpenGL symbols initially not present::


    [ 63%] Building CXX object CMakeFiles/OGLRap.dir/Renderer.cc.o
    [ 65%] Building CXX object CMakeFiles/OGLRap.dir/RContext.cc.o
    /home/blyth/opticks/oglrap/Prog.cc: In member function ‘void Prog::setup()’:
    /home/blyth/opticks/oglrap/Prog.cc:116:23: error: ‘GL_GEOMETRY_SHADER’ was not declared in this scope
         m_codes.push_back(GL_GEOMETRY_SHADER);
                           ^
    /home/blyth/opticks/oglrap/G.cc: In static member function ‘static const char* G::Shader(GLenum)’:
    /home/blyth/opticks/oglrap/G.cc:50:13: error: ‘GL_GEOMETRY_SHADER’ was not declared in this scope
            case GL_GEOMETRY_SHADER: s = GL_GEOMETRY_SHADER_ ; break ; 
                 ^
    /home/blyth/opticks/oglrap/G.cc: In static member function ‘static const char* G::Err(GLenum)’:
    /home/blyth/opticks/oglrap/G.cc:68:14: error: ‘GL_CONTEXT_LOST’ was not declared in this scope
             case GL_CONTEXT_LOST : s = GL_CONTEXT_LOST_ ; break ;
                  ^
    make[2]: *** [CMakeFiles/OGLRap.dir/G.cc.o] Error 1
    make[2]: *** Waiting for unfinished jobs....
    /home/blyth/opticks/oglrap/RContext.cc: In member function ‘void RContext::initUniformBuffer()’:
    /home/blyth/opticks/oglrap/RContext.cc:63:18: error: ‘GL_UNIFORM_BUFFER’ was not declared in this scope
         glBindBuffer(GL_UNIFORM_BUFFER, this->uniformBO);
                      ^
    make[2]: *** [CMakeFiles/OGLRap.dir/Prog.cc.o] Error 1
    /home/blyth/opticks/oglrap/RContext.cc: In member function ‘void RContext::bindUniformBlock(GLuint)’:
    /home/blyth/opticks/oglrap/RContext.cc:75:82: error: ‘glGetUniformBlockIndex’ was not declared in this scope
         GLuint uniformBlockIndex = glGetUniformBlockIndex(program,  uniformBlockName ) ;
                                                                                      ^
    In file included from /usr/include/c++/4.8.2/cassert:43:0,
                     from /home/blyth/local/opticks/externals/plog/include/plog/Util.h:2,
                     from /home/blyth/local/opticks/externals/plog/include/plog/Record.h:3,
                     from /home/blyth/local/opticks/externals/plog/include/plog/Appenders/IAppender.h:2,
                     from /home/blyth/local/opticks/externals/plog/include/plog/Logger.h:2,
                     from /home/blyth/local/opticks/externals/plog/include/plog/Log.h:7,
                     from /home/blyth/local/opticks/include/SysRap/PLOG.hh:26,
                     from /home/blyth/opticks/oglrap/RContext.cc:26:
    /home/blyth/opticks/oglrap/RContext.cc:76:33: error: ‘GL_INVALID_INDEX’ was not declared in this scope
         assert(uniformBlockIndex != GL_INVALID_INDEX && "NB must use the uniform otherwise it gets optimized away") ;
                                     ^
    /home/blyth/opticks/oglrap/RContext.cc:78:76: error: ‘glUniformBlockBinding’ was not declared in this scope
         glUniformBlockBinding(program, uniformBlockIndex,  uniformBlockBinding );
                                                                                ^
    /home/blyth/opticks/oglrap/RContext.cc: In member function ‘void RContext::update(const mat4&, const mat4&, const vec4&)’:
    /home/blyth/opticks/oglrap/RContext.cc:91:18: error: ‘GL_UNIFORM_BUFFER’ was not declared in this scope
         glBindBuffer(GL_UNIFORM_BUFFER, this->uniformBO);
                      ^
    /home/blyth/opticks/oglrap/InstLODCull.cc: In member function ‘void InstLODCull::applyFork()’:
    /home/blyth/opticks/oglrap/InstLODCull.cc:118:72: error: ‘glBeginQueryIndexed’ was not declared in this scope
             glBeginQueryIndexed(GL_PRIMITIVES_GENERATED, i, m_lodQuery[i]  );
                                                                            ^
    /home/blyth/opticks/oglrap/InstLODCull.cc:125:54: error: ‘glEndQueryIndexed’ was not declared in this scope
             glEndQueryIndexed(GL_PRIMITIVES_GENERATED, i );
                                                          ^
    /home/blyth/opticks/oglrap/InstLODCull.cc: In member function ‘void InstLODCull::applyForkStreamQueryWorkaround()’:
    /home/blyth/opticks/oglrap/InstLODCull.cc:174:72: error: ‘glBeginQueryIndexed’ was not declared in this scope
             glBeginQueryIndexed(GL_PRIMITIVES_GENERATED, i, m_lodQuery[i]  );
                                                                            ^
    /home/blyth/opticks/oglrap/InstLODCull.cc:178:54: error: ‘glEndQueryIndexed’ was not declared in this scope
             glEndQueryIndexed(GL_PRIMITIVES_GENERATED, i );
                                                          ^
    /home/blyth/opticks/oglrap/InstLODCull.cc: In member function ‘void InstLODCull::initShader()’:
    /home/blyth/opticks/oglrap/InstLODCull.cc:270:76: error: cannot convert ‘const char**’ to ‘const GLint* {aka const int*}’ in argument passing
         glTransformFeedbackVaryings(m_program, 14, vars, GL_INTERLEAVED_ATTRIBS);
                                                                                ^
    make[2]: *** [CMakeFiles/OGLRap.dir/RContext.cc.o] Error 1
    /home/blyth/opticks/oglrap/Rdr.cc: In member function ‘void Rdr::address(ViewNPY*)’:
    /home/blyth/opticks/oglrap/Rdr.cc:418:60: error: ‘GL_FIXED’ was not declared in this scope
             case ViewNPY::FIXED:                        type = GL_FIXED                        ; break ;
                                                                ^
    /home/blyth/opticks/oglrap/Rdr.cc:419:60: error: ‘GL_INT_2_10_10_10_REV’ was not declared in this scope
             case ViewNPY::INT_2_10_10_10_REV:           type = GL_INT_2_10_10_10_REV           ; break ; 
                                                                ^
    make[2]: *** [CMakeFiles/OGLRap.dir/InstLODCull.cc.o] Error 1
    make[2]: *** [CMakeFiles/OGLRap.dir/Rdr.cc.o] Error 1
    /home/blyth/opticks/oglrap/Renderer.cc: In member function ‘GLuint Renderer::createVertexArray(RBuf*)’:
    /home/blyth/opticks/oglrap/Renderer.cc:486:54: error: ‘glVertexAttribDivisor’ was not declared in this scope
             glVertexAttribDivisor(vTransform + 0, divisor);  // dictates instanced geometry shifts between instances
                                                          ^
    /home/blyth/opticks/oglrap/Renderer.cc: In member function ‘void Renderer::render()’:
    /home/blyth/opticks/oglrap/Renderer.cc:640:104: error: ‘glDrawElementsInstanced’ was not declared in this scope
                 glDrawElementsInstanced( draw.mode, draw.count, draw.type,  draw.indices, m_lod_counts[i]  ) ;
                                                                                                            ^
    /home/blyth/opticks/oglrap/Renderer.cc:657:104: error: ‘glDrawElementsInstanced’ was not declared in this scope
                 glDrawElementsInstanced( draw.mode, draw.count, draw.type,  draw.indices, m_lod_counts[i]  ) ;
                                                                                                            ^
    /home/blyth/opticks/oglrap/Renderer.cc:668:103: error: ‘glDrawElementsInstanced’ was not declared in this scope
                 glDrawElementsInstanced( draw.mode, draw.count, draw.type,  draw.indices, draw.primcount  ) ;
                                                                                                           ^
    make[2]: *** [CMakeFiles/OGLRap.dir/Renderer.cc.o] Error 1
    make[1]: *** [CMakeFiles/OGLRap.dir/all] Error 2
    make: *** [all] Error 2
    === om-one-or-all make : non-zero rc 2
    === om-all om-make : ERROR bdir /home/blyth/local/opticks/build/oglrap : non-zero rc 2
    [blyth@localhost opticks]$ 
i



examples/UseOpticksGLEW also demonstrates the grabbing of wrong libGLEW.so::


    ====== tgt:Opticks::OpticksGLEW tgt_DIR: ================
    tgt='Opticks::OpticksGLEW' prop='INTERFACE_INCLUDE_DIRECTORIES' defined='0' set='1' value='/home/blyth/junotop/ExternalLibs/ROOT/6.18.00/include' 

    tgt='Opticks::OpticksGLEW' prop='INTERFACE_FIND_PACKAGE_NAME' defined='1' set='1' value='OpticksGLEW' 

    tgt='Opticks::OpticksGLEW' prop='IMPORTED_LOCATION' defined='0' set='1' value='/home/blyth/junotop/ExternalLibs/ROOT/6.18.00/lib/libGLEW.so' 


    -- Configuring done
    -- Generating done
    -- Build files have been written to: /tmp/blyth/opticks/UseOpticksGLEW/build
    Scanning dependencies of target UseOpticksGLEW
    [ 50%] Building CXX object CMakeFiles/UseOpticksGLEW.dir/UseOpticksGLEW.cc.o
    [100%] Linking CXX executable UseOpticksGLEW
    [100%] Built target UseOpticksGLEW
    [100%] Built target UseOpticksGLEW
    Install the project...
    -- Install configuration: "Debug"
    -- Installing: /home/blyth/local/opticks/lib/UseOpticksGLEW
    -- Set runtime path of "/home/blyth/local/opticks/lib/UseOpticksGLEW" to "$ORIGIN/../lib64:$ORIGIN/../externals/lib:$ORIGIN/../externals/lib64:$ORIGIN/../externals/OptiX/lib64:/home/blyth/junotop/ExternalLibs/ROOT/6.18.00/lib"
    GL_VERSION_1_1
    GL_VERSION_2_0
    GL_VERSION_3_0
    [blyth@localhost UseOpticksGLEW]$ om-export-info


After commenting the ROOT paths Can pickup the correct libGLEW::

    ====== tgt:Opticks::OpticksGLEW tgt_DIR: ================
    tgt='Opticks::OpticksGLEW' prop='INTERFACE_INCLUDE_DIRECTORIES' defined='0' set='1' value='/home/blyth/local/opticks/externals/include' 

    tgt='Opticks::OpticksGLEW' prop='INTERFACE_FIND_PACKAGE_NAME' defined='1' set='1' value='OpticksGLEW' 

    tgt='Opticks::OpticksGLEW' prop='IMPORTED_LOCATION' defined='0' set='1' value='/home/blyth/local/opticks/externals/lib/libGLEW.so' 


    -- Configuring done
    -- Generating done
    -- Build files have been written to: /tmp/blyth/opticks/UseOpticksGLEW/build
    Scanning dependencies of target UseOpticksGLEW
    [ 50%] Building CXX object CMakeFiles/UseOpticksGLEW.dir/UseOpticksGLEW.cc.o
    [100%] Linking CXX executable UseOpticksGLEW
    [100%] Built target UseOpticksGLEW
    [100%] Built target UseOpticksGLEW
    Install the project...
    -- Install configuration: "Debug"
    -- Installing: /home/blyth/local/opticks/lib/UseOpticksGLEW
    -- Set runtime path of "/home/blyth/local/opticks/lib/UseOpticksGLEW" to "$ORIGIN/../lib64:$ORIGIN/../externals/lib:$ORIGIN/../externals/lib64:$ORIGIN/../externals/OptiX/lib64:/home/blyth/local/opticks/externals/lib"
    GL_VERSION_1_1
    GL_VERSION_2_0
    GL_VERSION_3_0
    GL_VERSION_4_0
    GL_VERSION_4_5
    [blyth@localhost UseOpticksGLEW]$ 





