# === func-gen- : graphics/assimprap/assimprap fgp graphics/assimprap/assimprap.bash fgn assimprap fgh graphics/assimprap
assimprap-src(){      echo assimprap/assimprap.bash ; }
assimprap-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(assimprap-src)} ; }
assimprap-vi(){       vi $(assimprap-source) ; }
assimprap-usage(){ cat << EOU

AssimpRap
============

AssimpRap converts geometry and material property information 
into unencumbered (no ai/AssimpRap) GGeo instances.

Depends on my github fork of the Assimp 3D Asset Importer Library,
which adds handling of G4DAE extra material/surface properties.

* see AssimpRapTest.cc for a demo of geometry conversion 
* see ggeo-vi
* used by raytrace-



Windows MSVC
-------------

::

    $ AssimpRapTest.exe
    C:/usr/local/opticks/lib/AssimpRapTest.exe: error while loading shared libraries: ?: cannot open shared object file: No such file or directory


Copying the assimp .lib (from ../externals/lib) and .dll from (../externals/bin) 
into /c/usr/local/opticks/lib allows the test to run::

    ntuhep@ntuhep-PC MINGW64 /c/usr/local/opticks/lib
    $ ll assimp*
    -rw-r--r-- 1 ntuhep 197121   164518 Jun 23 16:31 assimp-vc100-mtd.lib
    -rwxr-xr-x 1 ntuhep 197121 13109760 Jun 23 16:33 assimp-vc100-mtd.dll*


Getting DAE for the test
----------------------------

::

    opticksdata-
    opticksdata-get
    opticksdata-export




Classes/compilation units
----------------------------

AssimpGeometry
    top level steering of the Assimp import of COLLADA G4DAE exported files

AssimpTree
    converts ai tree into a tree of AssimpNode with parent/child
    hookups matching the original Geant4 pv/lv tree 

AssimpNode
    holds mesh data and transforms

AssimpRegistry
    hashed map of AssimpNode used for parent/child hookup

AssimpSelection
    geometry selection query parsing and execution

AssimpGGeo
    converts an AssimpTree and AssimpSelection into 
    an unencumbered (no ai or AssimpRap) GGeo instance, see ggeo-vi

AssimpCommon
    common aiNode/aiMesh handling functions

md5digest
    hashing 


Coordinate Mismatch
--------------------

On attempting to visualize genstep data within the geometry 
find the event data coordinates are not within the geometry.
But the coordinates are related by a rotation.


::

   a.g4daeview.g4daeview:100 vbo.data array([ ([-16585.724609375, -802008.375, -3600.0]

   RendererBase::dump       0/5475634 : world   -16585.7    -3600.0   802008.4

   Theres a rotation about X causing,  (x,y,z) => (x,z,-y )

              1  0  0
              0  0  1
              0 -1  0


       -Y    Z                           
          .  |                     
           . |                       
            .|                      
     X ------+          
              \
               \ 
                Y


Turns out assimp is obeying rotating according to the up direction specified 
in the DAE file. But there is an import switch to ignore this::

    delta:assimp-fork blyth$ find . -type f -exec grep -H _COLLADA_IGNORE_UP_DIRECTION {} \;
    ./code/ColladaLoader.cpp:   ignoreUpDirection = pImp->GetPropertyInteger(AI_CONFIG_IMPORT_COLLADA_IGNORE_UP_DIRECTION,0) != 0;
    ./include/assimp/config.h:#define AI_CONFIG_IMPORT_COLLADA_IGNORE_UP_DIRECTION "IMPORT_COLLADA_IGNORE_UP_DIRECTION"

Switch this off in AssimpGeometry.cc::

     m_importer->SetPropertyInteger(AI_CONFIG_IMPORT_COLLADA_IGNORE_UP_DIRECTION,1);



Windows Runtime Launch Failures 
------------------------------------

::

    ntuhep@ntuhep-PC MINGW64 ~/env/graphics/assimprap/tests
    $ AssimpWrapTest.exe
    C:/msys64/usr/local/opticks/bin/AssimpWrapTest.exe: error while loading shared libraries: libGGeo.dll: cannot open shared object file: No such file or directory

Lots of absentees in ldd::

    $ ldd $(which AssimpWrapTest.exe) | grep opticks
            libAssimpRap.dll => /usr/local/opticks/lib/libAssimpRap.dll (0x62000000)

After adjust assimp-prefix to use a common location and include that in PATH get::

    $ ldd $(which AssimpWrapTest.exe) | grep opticks
            libAssimpRap.dll => /usr/local/opticks/lib/libAssimpRap.dll (0x62000000)
            libassimp.dll => /usr/local/opticks/externals/bin/libassimp.dll (0x540000)
            libOpticksCore.dll => /usr/local/opticks/lib/libOpticksCore.dll (0xd90000)
            libBCfg.dll => /usr/local/opticks/lib/libBCfg.dll (0x65180000)
            libBRegex.dll => /usr/local/opticks/lib/libBRegex.dll (0x6cbc0000)
            libNPY.dll => /usr/local/opticks/lib/libNPY.dll (0x30c0000)
            libGGeo.dll => /usr/local/opticks/lib/libGGeo.dll (0x69740000)

And it runs::

    $ AssimpWrapTest.exe
    Opticks::preargs argc 1
    [2016-06-06 20:34:22.972810] [0x00000694] [info]    Opticks:: START
    OpticksResource::readEnvironment USING DEFAULT geokey DAE_NAME_DYB
    OpticksResource::readEnvironment MISSING ENVVAR pointing to geometry for geokey DAE_NAME_DYB path (null)
    OpticksResource::readEnvironment USING DEFAULT geo query range:3153:12221
    [2016-06-06 20:34:22.972810] [0x00000694] [info]    OpticksQuery::parseQuery query:[range:3153:12221] elements:1 queryType:range
    [2016-06-06 20:34:22.972810] [0x00000694] [info]    OpticksQuery::init dumpQuery queryType range m_query_string range:3153:12221 m_query_name NULL m_query_index 0 nrange 2 : 3153 : 12221

    OpticksResource::readEnvironment USING DEFAULT geo ctrl volnames

    This application has requested the Runtime to terminate it in an unusual way.
    Please contact the application's support team for more information.
    after ok
    Assertion failed!

    Program: C:\msys64\usr\local\opticks\bin\AssimpWrapTest.exe
    File: C:/msys64/mingw64/include/boost/filesystem/path_traits.hpp, Line 331

    Expression: c_str
     

Note:

* **failed launch due to failure to find lib/dll error messages are abysmal**




Workflow
---------

::

   assimprap-extra



FUNCTIONS
----------

Border Surfaces Looking Reasonable with expected AD symmetry
---------------------------------------------------------------

::

    AssimpGGeo::convertMaterials materialIndex 1
        bspv1 __dd__Geometry__AdDetails__lvBotReflector--pvBotRefGap0xbfa6458
        bspv2 __dd__Geometry__AdDetails__lvBotRefGap--pvBotESR0xbf9bd08 
    AssimpGGeo::convertMaterials materialIndex 2
        bspv1 __dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xc266468
        bspv2 __dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xc4110d0 
    AssimpGGeo::convertMaterials materialIndex 4
        bspv1 __dd__Geometry__AD__lvSST--pvOIL0xc241510
        bspv2 __dd__Geometry__AD__lvADE--pvSST0xc128d90 
    AssimpGGeo::convertMaterials materialIndex 5
        bspv1 __dd__Geometry__Pool__lvNearPoolIWS--pvNearADE10xc2cf528
        bspv2 __dd__Geometry__AD__lvADE--pvSST0xc128d90 
    AssimpGGeo::convertMaterials materialIndex 6
        bspv1 __dd__Geometry__Pool__lvNearPoolIWS--pvNearADE20xc0479c8
        bspv2 __dd__Geometry__AD__lvADE--pvSST0xc128d90 
    AssimpGGeo::convertMaterials materialIndex 7
        bspv1 __dd__Geometry__Sites__lvNearHallBot--pvNearPoolDead0xc13c018
        bspv2 __dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner0xbf4b270 
    AssimpGGeo::convertMaterials materialIndex 8
        bspv1 __dd__Geometry__Pool__lvNearPoolCurtain--pvNearPoolIWS0xc15a498
        bspv2 __dd__Geometry__Pool__lvNearPoolOWS--pvNearPoolCurtain0xc5c5f20 
    AssimpGGeo::convertMaterials materialIndex 11
        bspv1 __dd__Geometry__Pool__lvNearPoolLiner--pvNearPoolOWS0xbf55b10
        bspv2 __dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner0xbf4b270 

    AssimpGGeo::convertStructureVisit border surface

    obs# 0 nodeIndex 3149 obs 0x11c2b2750 idx  7
        pv_p __dd__Geometry__Sites__lvNearHallBot--pvNearPoolDead0xc13c018
        pv   __dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner0xbf4b270
    ibs# 0 nodeIndex 3150 ibs 0x11c2b3370 idx 11
        pv   __dd__Geometry__Pool__lvNearPoolLiner--pvNearPoolOWS0xbf55b10
        pv_p __dd__Geometry__Pool__lvNearPoolDead--pvNearPoolLiner0xbf4b270
    ibs# 1 nodeIndex 3152 ibs 0x11c2b2b00 idx  8
        pv   __dd__Geometry__Pool__lvNearPoolCurtain--pvNearPoolIWS0xc15a498
        pv_p __dd__Geometry__Pool__lvNearPoolOWS--pvNearPoolCurtain0xc5c5f20

    obs# 1 nodeIndex 3154 obs 0x11c2b23d0 idx  5
        pv_p __dd__Geometry__Pool__lvNearPoolIWS--pvNearADE10xc2cf528
        pv   __dd__Geometry__AD__lvADE--pvSST0xc128d90
    ibs# 2 nodeIndex 3155 ibs 0x11c2b2220 idx  4
        pv   __dd__Geometry__AD__lvSST--pvOIL0xc241510
        pv_p __dd__Geometry__AD__lvADE--pvSST0xc128d90
    obs# 2 nodeIndex 4427 obs 0x11c2b1bc0 idx  2
        pv_p __dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xc266468
        pv   __dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xc4110d0
    obs# 3 nodeIndex 4430 obs 0x1061cc450 idx  1
        pv_p __dd__Geometry__AdDetails__lvBotReflector--pvBotRefGap0xbfa6458
        pv   __dd__Geometry__AdDetails__lvBotRefGap--pvBotESR0xbf9bd08

    obs# 4 nodeIndex 4814 obs 0x11c2b2530 idx  6
        pv_p __dd__Geometry__Pool__lvNearPoolIWS--pvNearADE20xc0479c8
        pv   __dd__Geometry__AD__lvADE--pvSST0xc128d90
    ibs# 3 nodeIndex 4815 ibs 0x11c2b2220 idx  4
        pv   __dd__Geometry__AD__lvSST--pvOIL0xc241510
        pv_p __dd__Geometry__AD__lvADE--pvSST0xc128d90
    obs# 5 nodeIndex 6087 obs 0x11c2b1bc0 idx  2
        pv_p __dd__Geometry__AdDetails__lvTopReflector--pvTopRefGap0xc266468
        pv   __dd__Geometry__AdDetails__lvTopRefGap--pvTopESR0xc4110d0
    obs# 6 nodeIndex 6090 obs 0x1061cc450 idx  1
        pv_p __dd__Geometry__AdDetails__lvBotReflector--pvBotRefGap0xbfa6458
        pv   __dd__Geometry__AdDetails__lvBotRefGap--pvBotESR0xbf9bd08



EOU
}
assimprap-env(){      olocal- ; opticks- ;  }

assimprap-sdir(){ echo $(opticks-home)/assimprap ; }
assimprap-tdir(){ echo $(opticks-home)/assimprap/tests ; }
assimprap-idir(){ echo $(opticks-idir); }
assimprap-bdir(){ echo $(opticks-bdir)/assimprap ; }

assimprap-icd(){  cd $(assimprap-idir); }
assimprap-bcd(){  cd $(assimprap-bdir); }
assimprap-scd(){  cd $(assimprap-sdir); }
assimprap-tcd(){  cd $(assimprap-tdir); }

assimprap-cd(){  cd $(assimprap-sdir); }

assimprap-wipe(){
    local bdir=$(assimprap-bdir)
    rm -rf $bdir
}


assimprap-name(){ echo AssimpRap ; }
assimprap-tag(){  echo ASIRAP ; }


assimprap--(){        opticks--     $(assimprap-bdir) ; }
assimprap-ctest(){    opticks-ctest $(assimprap-bdir) $* ; }
assimprap-genproj(){  assimprap-scd ; opticks-genproj $(assimprap-name) $(assimprap-tag) ; }
assimprap-gentest(){  assimprap-tcd ; opticks-gentest ${1:-AssimpGGeo} $(assimprap-tag) ; }
assimprap-txt(){ vi $(assimprap-sdir)/CMakeLists.txt $(assimprap-tdir)/CMakeLists.txt ; } 




