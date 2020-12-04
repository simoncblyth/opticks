NPY-remove-deps-OpenMesh-ImplicitMesher
=========================================

Whilst attempting to remove deps in NPY encounter header finding problem with NGLTF.cpp::

     21 #include <sstream>
     22 
     23 #include "PLOG.hh"
     24 #include "BFile.hh"
     25 
     26 #include "NYGLTF.hpp"
     27 #include "NGLTF.hpp"

NYGLTF.hpp::

     34 
     35 #include "YoctoGL/yocto_gltf.h"
     36 


Reinstated all deps to see the successful commandline by touch and VERBOSE::

    epsilon:npy blyth$ touch NGLTF.cpp ; VERBOSE=1 om 

    [  0%] Building CXX object CMakeFiles/NPY.dir/NGLTF.cpp.o
    /Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  
        -DBOOST_ALL_NO_LIB -DBOOST_FILESYSTEM_DYN_LINK -DBOOST_PROGRAM_OPTIONS_DYN_LINK -DBOOST_REGEX_DYN_LINK -DBOOST_SYSTEM_DYN_LINK -DNPY_EXPORTS -DOPTICKS_BRAP -DOPTICKS_DualContouringSample -DOPTICKS_ImplicitMesher -DOPTICKS_NPY -DOPTICKS_OKCONF -DOPTICKS_OpenMesh -DOPTICKS_SYSRAP -DOPTICKS_YoctoGL -DWITH_BOOST_ASIO 
    -I/Users/blyth/opticks/npy 
    -isystem /usr/local/opticks/externals/glm/glm 
    -isystem /usr/local/opticks/include/SysRap 
    -isystem /usr/local/opticks/externals/plog/include 
    -isystem /usr/local/opticks/include/OKConf 
    -isystem /usr/local/opticks/include/BoostRap 
    -isystem /usr/local/opticks_externals/boost/include 
    -isystem /usr/local/opticks/externals/include 
    -isystem /usr/local/opticks/externals/include/YoctoGL 
    -isystem /usr/local/opticks/externals/include/ImplicitMesher 
    -isystem /usr/local/opticks/externals/include/DualContouringSample  
    -fvisibility=hidden 
    -fvisibility-inlines-hidden 
    -fdiagnostics-show-option 
    -Wall -Wno-unused-function -Wno-unused-private-field -Wno-shadow -g -fPIC   -std=gnu++14 -o CMakeFiles/NPY.dir/NGLTF.cpp.o -c /Users/blyth/opticks/npy/NGLTF.cpp
    [  1%] Linking CXX shared library libNPY.dylib



Seems that are relying on the indiscriminate::

    -isystem /usr/local/opticks/externals/include 

Rather than::

    -isystem /usr/local/opticks/externals/include/YoctoGL 

And somehow skipping the deps prevents that indiscriminate header path being used ?

::

    epsilon:npy blyth$ opticks-externals
    bcm
    glm
    glfw
    glew
    gleq
    imgui
    openmesh
    plog
    opticksaux
    oimplicitmesher
    odcs
    oyoctogl
    ocsgbsp
    epsilon:npy blyth$ 


oimplicitmesher-cd ; vi CMakeLists.txt::

     01 cmake_minimum_required (VERSION 3.5)
      2 set(name ImplicitMesher)
      3 project(${name} VERSION 0.1.0)
      4 include(OpticksBuildOptions)
      5 
      6 #[=[
      7 Hmm OpticksBuildOptions sets CMAKE_INSTALL_INCLUDEDIR to "include/${name}"
      8 so must override that here rather than from commandline
      9 #]=]
     10 
     11 include(GNUInstallDirs)
     12 set(CMAKE_INSTALL_INCLUDEDIR "externals/include/${name}")
     13 set(CMAKE_INSTALL_LIBDIR     "externals/lib")
     14 set(CMAKE_INSTALL_BINDIR     "lib")
     15 
     ..
     76 
     77 bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)
     78 install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})


openmesh-cd ; vi CMakeLists.txt

Hmm : actually it is not the CMakeLists.txt of the externals that matters, but rather
how they are found.


cmake/Modules/FindOpenMesh.cmake::

     12 set(OpenMesh_PREFIX "${OPTICKS_PREFIX}/externals")
     13 
     14 find_path( OpenMesh_INCLUDE_DIR
     15            NAMES "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
     16            PATHS "${OpenMesh_PREFIX}/include"
     17 )
     18 


Looks like OpenMesh_INCLUDE_DIR is the culprit unspecific.

cd ~/opticks/examples/UseOpenMesh::

   ./go.sh 
   ...
   -- FindOpenMesh.cmake OpenMesh_MODULE     :/Users/blyth/opticks/cmake/Modules/FindOpenMesh.cmake  
   -- FindOpenMesh.cmake OpenMesh_INCLUDE_DIR:/usr/local/opticks/externals/include  
   ...



Rejig to make OpenMesh_INCLUDE_DIR end with OpenMesh::

     14 find_path( OpenMesh_INCLUDE_DIR
     15            NAMES "OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh"
     16            PATHS "${OpenMesh_PREFIX}/include"
     17 )
     18 

     14 find_path( OpenMesh_INCLUDE_DIR
     15            NAMES "Core/Mesh/TriMesh_ArrayKernelT.hh"
     16            PATHS "${OpenMesh_PREFIX}/include/OpenMesh"
     17 )

Doesnt work as inclusion is expecting the OpenMesh::

    [ 50%] Building CXX object CMakeFiles/UseOpenMesh.dir/UseOpenMesh.cc.o
    /Users/blyth/opticks/examples/UseOpenMesh/UseOpenMesh.cc:22:10: fatal error: 'OpenMesh/Core/IO/MeshIO.hh' file not found
    #include <OpenMesh/Core/IO/MeshIO.hh>
             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1 error generated.
    make[2]: *** [CMakeFiles/UseOpenMesh.dir/UseOpenMesh.cc.o] Error 1
    make[1]: *** [CMakeFiles/UseOpenMesh.dir/all] Error 2
    make: *** [all] Error 2
    [ 50%] Building CXX object CMakeFiles/UseOpenMesh.dir/UseOpenMesh.cc.o
    /Users/blyth/opticks/examples/UseOpenMesh/UseOpenMesh.cc:22:10: fatal error: 'OpenMesh/Core/IO/MeshIO.hh' file not found
    #include <OpenMesh/Core/IO/MeshIO.hh>
             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1 error generated.


Seems its a necessary evil, so prevent its presence being dependent on the current deps and do it always.

npy/CMakeLists.txt::

    455 # some header inclusion expects the package name prefix, eg OpenMesh YoctoGL see notes/issues/NPY-remove-deps-OpenMesh-ImplicitMesher.rst
    456 target_include_directories( ${name} PUBLIC ${OPTICKS_PREFIX}/externals/include )
    45


Two test fails from removing the deps::

    FAILS:  2   / 444   :  Sat Dec  5 00:36:49 2020   
      44 /57  Test #44 : GGeoTest.GMakerTest                           Child aborted***Exception:     4.53   
      2  /2   Test #2  : IntegrationTests.tboolean.box                 ***Failed                      8.36   
    [blyth@localhost opticks]$ 
        
Avoid GMakerTest fail using WITH_OPENMESH to skip GMakerTest::makeFromCSG.

Check integration with::

    [blyth@localhost tests]$ LV=box tboolean.sh --generateoverride 10000 -D 
    ...

    2020-12-05 00:38:35.635 INFO  [272787] [OpticksHub::setupTestGeometry@355] --test modifying geometry
    2020-12-05 00:38:35.639 FATAL [272787] [NCSG::polygonize@1130] NCSG::polygonize requires compilation with the optional OpenMesh
    OKG4Test: /home/blyth/opticks/npy/NCSG.cpp:1131: NTrianglesNPY* NCSG::polygonize(): Assertion `0' failed.

    Program received signal SIGABRT, Aborted.
    0x00007fffe519f387 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install bzip2-libs-1.0.6-13.el7.x86_64 cyrus-sasl-lib-2.1.26-23.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 freetype-2.8-12.el7_6.1.x86_64 glibc-2.17-307.el7.1.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libICE-1.0.9-9.el7.x86_64 libSM-1.2.2-2.el7.x86_64 libX11-1.6.7-2.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libcurl-7.29.0-57.el7.x86_64 libgcc-4.8.5-39.el7.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libidn-1.28-4.el7.x86_64 libpng-1.5.13-7.el7_2.x86_64 libselinux-2.5-14.1.el7.x86_64 libssh2-1.8.0-3.el7.x86_64 libstdc++-4.8.5-39.el7.x86_64 libuuid-2.23.2-59.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 nspr-4.19.0-1.el7_5.x86_64 nss-3.36.0-7.1.el7_6.x86_64 nss-softokn-freebl-3.36.0-5.el7_5.x86_64 nss-util-3.36.0-1.1.el7_6.x86_64 openldap-2.4.44-21.el7_6.x86_64 openssl-libs-1.0.2k-19.el7.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffe519f387 in raise () from /lib64/libc.so.6
    #1  0x00007fffe51a0a78 in abort () from /lib64/libc.so.6
    #2  0x00007fffe51981a6 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe5198252 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffec5bb2fb in NCSG::polygonize (this=0x71f45d0) at /home/blyth/opticks/npy/NCSG.cpp:1131
    #5  0x00007fffed679f8a in GMaker::makeMeshFromCSG (this=0x71f3d80, csg=0x71f45d0) at /home/blyth/opticks/ggeo/GMaker.cc:169
    #6  0x00007fffed676e69 in GGeoTest::prepareMeshes (this=0x71eb320) at /home/blyth/opticks/ggeo/GGeoTest.cc:492
    #7  0x00007fffed6767ee in GGeoTest::importCSG (this=0x71eb320) at /home/blyth/opticks/ggeo/GGeoTest.cc:379
    #8  0x00007fffed67636f in GGeoTest::initCreateCSG (this=0x71eb320) at /home/blyth/opticks/ggeo/GGeoTest.cc:283
    #9  0x00007fffed675a03 in GGeoTest::init (this=0x71eb320) at /home/blyth/opticks/ggeo/GGeoTest.cc:177
    #10 0x00007fffed67574e in GGeoTest::GGeoTest (this=0x71eb320, ok=0x6d31a0, basis=0x70a060) at /home/blyth/opticks/ggeo/GGeoTest.cc:162
    #11 0x00007fffed94af46 in OpticksHub::setupTestGeometry (this=0x6f5850) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:361
    #12 0x00007fffed94a9c0 in OpticksHub::loadGeometry (this=0x6f5850) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:297
    #13 0x00007fffed94a53d in OpticksHub::init (this=0x6f5850) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:248
    #14 0x00007fffed94a232 in OpticksHub::OpticksHub (this=0x6f5850, ok=0x6d31a0) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:215
    #15 0x00007ffff7baf25c in OKG4Mgr::OKG4Mgr (this=0x7fffffff68a0, argc=32, argv=0x7fffffff6be8) at /home/blyth/opticks/okg4/OKG4Mgr.cc:100
    #16 0x000000000040393a in main (argc=32, argv=0x7fffffff6be8) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:27
    (gdb) 



::

    372 GVolume* GGeoTest::importCSG()
    373 {
    374     LOG(LEVEL) << "[" ;
    375     m_mlib->addTestMaterials();
    376 
    377     reuseMaterials(m_csglist);
    378   
    379     prepareMeshes();
    380 
    381     adjustContainer();
    382 
    383     int primIdx(-1) ;
    384 
    385     GVolume* top = NULL ;
    386     GVolume* prior = NULL ;
    387 
    388     unsigned num_mesh = m_meshlib->getNumMeshes();
    389 
    390     for(unsigned i=0 ; i < num_mesh ; i++)
    391     {
    392         primIdx++ ; // each tree is separate OptiX primitive, with own line in the primBuffer 
    393 
    394         GMesh* mesh = m_meshlib->getMeshSimple(i);
    395 
    396         unsigned ndIdx = i ;
    397         GVolume* volume = m_maker->makeVolumeFromMesh(ndIdx, mesh);
    398         if( top == NULL ) top = volume ;
    399 
    400         if(prior)
    401         {
    402             volume->setParent(prior);
    403             prior->addChild(volume);
    404         }
    405         prior = volume ;




    472 /**
    473 GGeoTest::prepareMeshes
    474 ------------------------------
    475 
    476 Proxied in geometry is centered
    477 
    478 **/
    479 
    480 void GGeoTest::prepareMeshes()
    481 {
    482     LOG(LEVEL) << "[" ;
    483 
    484     assert(m_csgpath);
    485     assert(m_csglist);
    486     //unsigned numTree = m_csglist->getNumTrees() ;
    487 
    488     assert( m_numtree > 0 );
    489     for(unsigned i=0 ; i < m_numtree ; i++)
    490     {
    491         NCSG* tree = m_csglist->getTree(i) ;
    492         GMesh* mesh =  tree->isProxy() ? importMeshViaProxy(tree) : m_maker->makeMeshFromCSG(tree) ;
    493         const char* name = BStr::concat<unsigned>("testmesh", i, NULL );
    494         mesh->setName(name);
    495 
    496         if(m_dbggeotest)
    497             mesh->Summary("GGeoTest::prepareMeshes");
    498 
    499         mesh->setIndex(m_meshlib->getNumMeshes());   // <-- used for GPt reference into GMeshLib.m_meshes
    500         m_meshlib->add(mesh);
    501     }
    502 
    503     LOG(LEVEL)
    504         << "]"
    505         << " csgpath " << m_csgpath
    506         << " m_numtree " << m_numtree
    507         << " verbosity " << m_verbosity
    508         ;
    509 }



    154 /**
    155 GMaker::makeMeshFromCSG
    156 ----------------------
    157 
    158 Hmm : this is using my (very temperamental) polygonization,
    159 but there is no need to do so in direct workflow as the Geant4 
    160 polygonization GMesh is available. 
    161 
    162 **/
    163 
    164 
    165 GMesh* GMaker::makeMeshFromCSG( NCSG* csg ) // cannot be const due to lazy NCSG::polgonize 
    166 {
    167     unsigned index = csg->getIndex();
    168     const char* spec = csg->getBoundary();
    169     NTrianglesNPY* tris = csg->polygonize();
    170 
    171     LOG(LEVEL)
    172         << " index " << index
    173         << " spec " << spec
    174         << " numTris " << ( tris ? tris->getNumTriangles() : 0 )
    175         << " trisMsg " << ( tris ? tris->getMessage() : "" )
    176         ;
    177 
    178     GMesh* mesh = GMeshMaker::Make(tris->getTris(), index);
    179     mesh->setCSG(csg);
    180     return mesh ;
    181 }


Perhaps just use bbox placeholder when no OpenMesh like "--x4polyskip"::


     57 GMesh* X4Mesh::Placeholder(const G4VSolid* solid ) //static
     58 {
     59 
     60 /*
     61     G4VisExtent ve = solid->GetExtent();
     62     //LOG(info) << " visExtent " << ve ; 
     63  
     64     nbbox bb = make_bbox( 
     65                    ve.GetXmin(), ve.GetYmin(), ve.GetZmin(), 
     66                    ve.GetXmax(), ve.GetYmax(), ve.GetZmax(),  false );  
     67 */
     68 
     69     nbbox* bb = X4SolidExtent::Extent(solid) ;
     70 
     71     NTrianglesNPY* tris = NTrianglesNPY::box(*bb) ;
     72     GMesh* mesh = GMeshMaker::Make(tris->getTris());
     73 




