geocache-cluster-fails-GGeo-trying-to-load-dae
==================================================

hmm not a very good response to no geocache for the key 
----------------------------------------------------------

* its a kinda mingling of the legacy geometry and the direct geometry
* need a clear distinction between these, for now based on "--envkey"
  once flip to that being the default can base on "--noenvkey" 

  * when envkey is in use can assert that the OPTICKS_KEY derived directory must exist 
     


::

    geocache-bench- is a function
    geocache-bench- () 
    { 
        type $FUNCNAME;
        local dbg;
        [ -n "$DBG" ] && dbg="gdb --args" || dbg="";
        $dbg OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig "steps=5,eyestartz=-1,eyestopz=-0.5" --size 5120,2880,1 --embedded $*
    }
    2019-04-28 21:24:19.703 INFO  [283514] [BOpticksKey::SetKey@45] from OPTICKS_KEY envvar OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
    2019-04-28 21:24:19.713 ERROR [283514] [OpticksResource::initRunResultsDir@262] /tmp/blyth/opticks/results/geocache-cluster/R0_cvd_0/20190428_212419
    2019-04-28 21:24:19.714 WARN  [283514] [OpticksHub::configure@296] OpticksHub::configure FORCED COMPUTE MODE : as remote session detected 
    2019-04-28 21:24:19.714 INFO  [283514] [OpticksHub::loadGeometry@480] [ /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe846377973af10a/1
    2019-04-28 21:24:19.721 ERROR [283514] [GGeo::loadFromG4DAE@618] GGeo::loadFromG4DAE START
    2019-04-28 21:24:19.721 INFO  [283514] [AssimpGGeo::load@143] AssimpGGeo::load  path NULL query all ctrl NULL importVerbosity 0 loaderVerbosity 0
    2019-04-28 21:24:19.721 FATAL [283514] [AssimpGGeo::load@155]  missing G4DAE path (null)
    2019-04-28 21:24:19.721 FATAL [283514] [GGeo::loadFromG4DAE@623] GGeo::loadFromG4DAE FAILED : probably you need to download opticksdata 
    OpSnapTest: /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GGeo.cc:627: void GGeo::loadFromG4DAE(): Assertion `rc == 0 && "G4DAE geometry file does not exist, try : opticksdata- ; opticksdata-- "' failed.
    /afs/ihep.ac.cn/users/b/blyth/g/opticks/ana/geocache.bash: line 329: 283514 Aborted                 (core dumped) $dbg OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig "steps=5,eyestartz=-1,eyestopz=-0.5" --size 5120,2880,1 --embedded $*
    Namespace(base='/afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-cluster', exclude=None, include=None)
    /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/results/geocache-cluster
    rc 0
    gpu019.ihep.ac.cn
    blyth@lxslc702~/g/opticks echo $OPTICKS_KEY
    OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
    blyth@lxslc702~/g/opticks 


Absolutely no direct geocache::

    blyth@lxslc702~/g/opticks l /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/
    total 56
    drwxr-xr-x  2 blyth dyw 28672 Apr 28 21:16 lib
    drwxr-xr-x  4 blyth dyw  4096 Apr 28 21:16 lib64
    drwxr-xr-x  5 blyth dyw  4096 Apr 28 20:36 installcache
    drwxr-xr-x 23 blyth dyw  4096 Apr 28 19:00 include
    drwxr-xr-x 23 blyth dyw  4096 Apr 28 19:00 build
    drwxr-xr-x 15 blyth dyw  4096 Apr 28 18:55 gl
    drwxr-xr-x 25 blyth dyw  4096 Apr 28 18:24 externals
    drwxr-xr-x  8 blyth dyw  4096 Apr 28 14:48 opticksdata
    blyth@lxslc702~/g/opticks 


