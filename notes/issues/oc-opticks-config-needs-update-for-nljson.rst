oc-opticks-config-needs-update-for-nljson
==========================================


issue non-cmake pkg-config tree needs updating for opticks package changes
------------------------------------------------------------------------------

::

    epsilon:~ blyth$ pkg-config --cflags SysRap
    -DOPTICKS_SYSRAP -DOPTICKS_OKCONF -I/usr/local/opticks/include/SysRap -I/usr/local/opticks/include/OKConf -I/usr/local/opticks/externals/plog/include

    epsilon:~ blyth$ pkg-config --cflags NPY
    Package NLJSON was not found in the pkg-config search path.
    Perhaps you should add the directory containing `NLJSON.pc'
    to the PKG_CONFIG_PATH environment variable
    Package 'NLJSON', required by 'BoostRap', not found

    epsilon:~ blyth$ oc-pkg-config-path-
    /usr/local/opticks_externals/g4_1042/lib/pkgconfig
    /usr/local/opticks_externals/clhep/lib/pkgconfig
    /usr/local/opticks_externals/xercesc/lib/pkgconfig
    /usr/local/opticks_externals/boost/lib/pkgconfig
    /usr/local/opticks/lib/pkgconfig
    /usr/local/opticks/externals/lib/pkgconfig

    epsilon:~ blyth$ l /usr/local/opticks/externals/lib/pkgconfig/
    total 104
    -rw-r--r--  1 blyth  staff  321 Sep 16 11:12 OpenMesh.pc
    -rw-r--r--  1 blyth  staff  468 May 12  2020 OptiX.pc
    -rw-r--r--  1 blyth  staff  377 May 12  2020 OpticksCUDA.pc
    -rw-r--r--  1 blyth  staff  253 May  7  2020 ImGui.pc
    -rw-r--r--  1 blyth  staff  330 May  7  2020 DualContouringSample.pc
    -rw-r--r--  1 blyth  staff  355 May  5  2020 OpticksGLEW.pc
    -rw-r--r--  1 blyth  staff  312 May  5  2020 PLog.pc
    -rw-r--r--  1 blyth  staff  414 May  5  2020 OpticksAssimp.pc
    -rw-r--r--  1 blyth  staff  476 May  5  2020 OpticksGLFW.pc
    -rw-r--r--  1 blyth  staff  380 May  5  2020 GLM.pc
    -rw-r--r--  1 blyth  staff  312 May  5  2020 ImplicitMesher.pc
    -rw-r--r--  1 blyth  staff  274 May  5  2020 CSGBSP.pc
    -rw-r--r--  1 blyth  staff  297 Apr 11  2020 YoctoGL.pc

    epsilon:~ blyth$ cat /usr/local/opticks/lib/pkgconfig/NPY.pc
    # bcm_auto_pkgconfig_each

    prefix=/usr/local/opticks
    exec_prefix=${prefix}
    libdir=${exec_prefix}/lib
    includedir=${exec_prefix}/include/NPY
    Name: npy
    Description: No description
    Version: 0.1.0

    Cflags:  -I/usr/local/opticks/externals/include -I${includedir} -DOPTICKS_NPY
    Libs: -L${libdir}  -lNPY
    Requires: GLM,SysRap,BoostRap,PLog


    epsilon:~ blyth$ cat /usr/local/opticks/lib/pkgconfig/BoostRap.pc 
    # bcm_auto_pkgconfig_each

    prefix=/usr/local/opticks
    exec_prefix=${prefix}
    libdir=${exec_prefix}/lib
    includedir=${exec_prefix}/include/BoostRap
    Name: boostrap
    Description: No description
    Version: 0.1.0

    Cflags:  -I${includedir} -DOPTICKS_BRAP -DWITH_BOOST_ASIO
    Libs: -L${libdir}  -lBoostRap
    Requires: PLog,SysRap,NLJSON


    epsilon:~ blyth$ opticks-externals
    bcm
    glm
    glfw
    glew
    gleq
    imgui
    plog
    opticksaux
    nljson


fixed by adding nljson-pc function and running it
----------------------------------------------------

::

    epsilon:~ blyth$  pkg-config --cflags NPY
    -DOPTICKS_NPY -DOPTICKS_BRAP -DWITH_BOOST_ASIO -DOPTICKS_SYSRAP -DOPTICKS_OKCONF -I/usr/local/opticks/externals/include -I/usr/local/opticks/include/NPY -I/usr/local/opticks/include/BoostRap -I/usr/local/opticks/include/SysRap -I/usr/local/opticks/include/OKConf -I/usr/local/opticks/externals/glm/glm -I/usr/local/opticks/externals/include/nljson -I/usr/local/opticks/externals/plog/include
    epsilon:~ blyth$ 

    epsilon:~ blyth$  pkg-config --cflags boostrap
    -DOPTICKS_BRAP -DWITH_BOOST_ASIO -DOPTICKS_SYSRAP -DOPTICKS_OKCONF -I/usr/local/opticks/include/BoostRap -I/usr/local/opticks/include/SysRap -I/usr/local/opticks/include/OKConf -I/usr/local/opticks/externals/plog/include -I/usr/local/opticks/externals/include/nljson
    epsilon:~ blyth$ 


    epsilon:~ blyth$  pkg-config --cflags G4OK
    -DOPTICKS_G4OK -DOPTICKS_CFG4 -DOPTICKS_X4 -DG4UI_USE_TCSH -W -Wall -pedantic -Wno-non-virtual-dtor -Wno-long-long -Wwrite-strings -Wpointer-arith -Woverloaded-virtual -Wno-variadic-macros -Wshadow -pipe -Qunused-arguments -stdlib=libc++ -DG4USE_STD11 -std=c++11 -DOPTICKS_OKOP -DOPTICKS_OXRAP -DOPTICKS_OKGEO -DOPTICKS_GGEO -DOPTICKS_THRAP -DOPTICKS_OKCORE -DOPTICKS_NPY -DOPTICKS_BRAP -DWITH_BOOST_ASIO -DOPTICKS_CUDARAP -DOPTICKS_SYSRAP -DOPTICKS_OKCONF -I/usr/local/opticks_externals/g4_1042/bin/../include/Geant4 -I/usr/local/opticks_externals/clhep/include -I/usr/local/opticks_externals/xercesc/include -I/usr/local/opticks/include/G4OK -I/usr/local/opticks/include/CFG4 -I/usr/local/opticks/include/ExtG4 -I/usr/local/opticks/include/OKOP -I/usr/local/opticks/include/OptiXRap -I/usr/local/opticks/include/OpticksGeo -I/usr/local/opticks/include/GGeo -I/usr/local/opticks/include/ThrustRap -I/usr/local/opticks/include/OpticksCore -I/usr/local/opticks/externals/include -I/usr/local/opticks/include/NPY -I/usr/local/opticks/include/BoostRap -I/usr/local/opticks/include/CUDARap -I/usr/local/opticks/include/SysRap -I/usr/local/opticks/include/OKConf -I/usr/local/optix/include -I/usr/local/opticks/externals/glm/glm -I/usr/local/opticks/externals/include/nljson -I/usr/local/opticks/externals/plog/include -I/usr/local/cuda/include
    epsilon:~ blyth$ 



