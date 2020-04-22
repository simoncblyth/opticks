oglrap-compilation-fail-junoenv-missing-new-opengl-symbols
==============================================================

JUNOTOP envvars causing old OpenGL symbols from some ROOT libGLEW.so

Fixed after manual edit of JUNOTOP/bashrc.sh excluding ROOT which brings in wrong libGLEW.so::

     13 source /home/blyth/junotop/ExternalLibs/CLHEP/2.4.1.0/bashrc
     14 source /home/blyth/junotop/ExternalLibs/xrootd/4.10.0/bashrc
     15 #source /home/blyth/junotop/ExternalLibs/ROOT/6.18.00/bashrc
     16 source /home/blyth/junotop/ExternalLibs/HepMC/2.06.09/bashrc
     17 source /home/blyth/junotop/ExternalLibs/Geant4/10.05.p01/bashrc


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





