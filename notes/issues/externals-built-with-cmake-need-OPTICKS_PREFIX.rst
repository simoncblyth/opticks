externals-built-with-cmake-need-OPTICKS_PREFIX
=================================================


Following cmake Find script changes to support user builds 
against binary releases it was necessary to change om-cmake
and om-cmake-okconf by adding the::

     -DOPTICKS_PREFIX=$(om-prefix) 

This is also needed by externals that use the Opticks
CMake Find scripts (FindGLM.cmake is a commonly used one)


::

    495 om-cmake()
    496 {
    497     local sdir=$1
    498     local bdir=$PWD
    499     [ "$sdir" == "$bdir" ] && echo ERROR sdir and bdir are the same $sdir && return 1000
    500 
    501     local rc
    502     cmake $sdir \
    503        -G "$(om-cmake-generator)" \
    504        -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
    505        -DOPTICKS_PREFIX=$(om-prefix) \
    506        -DCMAKE_PREFIX_PATH=$(om-prefix)/externals \
    507        -DCMAKE_INSTALL_PREFIX=$(om-prefix) \
    508        -DCMAKE_MODULE_PATH=$(om-home)/cmake/Modules
    509 
    510     rc=$?
    511     return $rc
    512 #   -DBOOST_INCLUDEDIR=$(opticks-boost-includedir) \
    513 #   -DBOOST_LIBRARYDIR=$(opticks-boost-libdir)
    514 }



::

    blyth@localhost externals]$ grep CMAKE_PREFIX_PATH *.bash
    g4.bash:    that is found by CMake via CMAKE_PREFIX_PATH::
    g4.bash:       -DCMAKE_PREFIX_PATH=$(om-prefix)/externals
    g4dae.bash:       -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
    imgui.bash:              -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
    ocsgbsp.bash:       -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
    odcs.bash:       -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
    oimplicitmesher.bash:       -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
    oyoctogl.bash:       -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
    
::

    [blyth@localhost externals]$ o
    M externals/imgui.bash
    M externals/ocsgbsp.bash
    M externals/odcs.bash
    M externals/oimplicitmesher.bash
    M externals/oyoctogl.bash
    ? notes/issues/externals-built-with-cmake-need-OPTICKS_PREFIX.rst
    [blyth@localhost opticks]
     

::

    [blyth@localhost opticks]$ opticks-externals
    bcm
    glm
    glfw
    glew
    gleq
    imgui
    assimp
    openmesh
    plog
    opticksaux
    oimplicitmesher
    odcs
    oyoctogl
    ocsgbsp
    xercesc
    g4


