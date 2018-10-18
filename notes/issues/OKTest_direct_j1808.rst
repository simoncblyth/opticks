OKTest direct j1808
=======================

Note j1808 is a transitional tag (it just corresponds to a path to a GDML file from Geant4 10.4.2 or later), 
as in the direct workflow there is no longer any need for such tags, or for the daepaths etc...
 
The geometry in direct mode is booted from the Geant4 10.4.2 GDML alone, and converted into GGeo geometry 
which is persisted to geocache into the keydir.  To run from that geocache it is necessary to note the
OPTICKS_KEY reported by the translation.

Opticks executables currently default to the old mode, to use direct geocahe use the "--envkey" argument
to make the executables sensitive to the OPTICKS_KEY envvar.

::

    [blyth@localhost 1]$ geocache-info

    OPTICKS_KEY     :  OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
    geocache-keydir : /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe846377973af10a/1
         directory derived from the OPTICKS_KEY envvar 


Creating the triangulated geocache::

    geocache-j1808 () 
    { 
        opticksdata-;
        OKX4Test --gdmlpath $(opticksdata-j) --g4codegen
    }


Visualizing the geocache::

    OKTest --envkey 
    ## raytrace is triangulated 

    OKTest --envkey --gltf 3
    ## mm0 asserts, currently need to select gltf at cache creation (TODO: avoid that ?)   


Creating the analytic geocache::

    geocache-j1808 () 
    { 
        opticksdata-;
        OKX4Test --gdmlpath $(opticksdata-j) --g4codegen --gltf 3  
    }


Nope gltf is old way of doing things it aint going to work in direct geometry route::

    2018-10-16 13:34:07.086 INFO  [106306] [BResource::Get@25]  label evtbase ret /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe846377973af10a/1/source
    2018-10-16 13:34:07.086 INFO  [106306] [BResource::Get@25]  label evtbase ret /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe846377973af10a/1/source
    2018-10-16 13:34:07.086 INFO  [106306] [Opticks::defineEventSpec@1285]  typ torch tag 1 det g4live cat 
    2018-10-16 13:34:07.087 FATAL [106306] [Opticks::configure@1379] Opticks::configure  m_size 1920,1080,1,0 m_position 100,100,0,0 prefdir $HOME/.opticks/g4live/State
    2018-10-16 13:34:07.087 ERROR [106306] [BFile::ExistsFile@265] BFile::ExistsFile BAD PATH path NULL sub NULL name NULL
    2018-10-16 13:34:07.087 FATAL [106306] [Opticks::configureCheckGeometryFiles@1122] gltf option is selected but there is no gltf file 
    2018-10-16 13:34:07.087 FATAL [106306] [Opticks::configureCheckGeometryFiles@1123]  SrcGLTFBase (null)
    2018-10-16 13:34:07.087 FATAL [106306] [Opticks::configureCheckGeometryFiles@1124]  SrcGLTFName (null)
    2018-10-16 13:34:07.087 FATAL [106306] [Opticks::configureCheckGeometryFiles@1125] Try to create the GLTF from GDML with eg:  op --j1707 --gdml2gltf  



Repeating below, surprised to find that get analytic raytrace, must have changed default OGeo geocode?::


    geocache-j1808 () 
    { 
        opticksdata-;
        OKX4Test --gdmlpath $(opticksdata-j) --g4codegen
    }


And in the new world order xanalytic is the way, Opticks::isXAnalytic() which sets the geocode on the geometry in OGeo::


    OKTest --envkey --xanalytic 

    ## visualizes the analytic geometry of the geocache identified by OPTICKS_KEY envvar    








