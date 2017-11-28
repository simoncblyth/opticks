axel-GSceneTest-fail
=====================


Expected way to make the analytic cache
------------------------------------------

::

    op --j1707 --gdml2gltf
       # convert the gdml into gltf with a python script

    op --j1707 -G
       # construct the triangulated geocache

    op --j1707 --gltf 3 -G
       # add analytic parts to the geocache



Observations
--------------

* Opticks::configureCheckGeometryFiles complaining about lack of 
  a different path than subsequently actually used ?



This is because of the argforced value 101::

    simon:ggeo blyth$ OpticksTest --gltf 101 2>&1 | cat |  grep GLTF
    2017-11-28 12:01:36.655 FATAL [30378] [Opticks::configureCheckGeometryFiles@830]  GLTFBase $TMP/nd
    2017-11-28 12:01:36.655 FATAL [30378] [Opticks::configureCheckGeometryFiles@831]  GLTFName scene.gltf
    2017-11-28 12:01:36.655 FATAL [30378] [Opticks::configureCheckGeometryFiles@832] Try to create the GLTF from GDML with eg:  op --j1707 --gdml2gltf  
                                   GLTFBase                                  $TMP/nd
                                   GLTFName                               scene.gltf
    simon:ggeo blyth$ 
    simon:ggeo blyth$ 
    simon:ggeo blyth$ OpticksTest --gltf 3 2>&1 | cat |  grep GLTF
                                   GLTFBase /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300
                                   GLTFName                               g4_00.gltf
    simon:ggeo blyth$ 



::

     798 const char* Opticks::getGLTFPath() const
     799 {
     800     return m_resource->getGLTFPath() ;
     801 }
     802 const char* Opticks::getGLTFBase() const  // config base and name only used whilst testing with gltf >= 100
     803 {
     804     int gltf = getGLTF();
     805     const char* path = getGLTFPath() ;
     806     std::string base = gltf < 100 ? BFile::ParentDir(path) : m_cfg->getGLTFBase() ;
     807     return strdup(base.c_str()) ;
     808 }
     809 const char* Opticks::getGLTFName() const
     810 {
     811     int gltf = getGLTF();
     812     const char* path = getGLTFPath() ;
     813     std::string name = gltf < 100 ? BFile::Name(path) : m_cfg->getGLTFName()  ;
     814     return strdup(name.c_str()) ;
     815 }
     816 



::

     649 void GGeo::loadAnalyticFromCache()
     650 {
     651     LOG(info) << "GGeo::loadAnalyticFromCache START" ;
     652     m_gscene = GScene::Load(m_ok, this); // GGeo needed for m_bndlib 
     653     LOG(info) << "GGeo::loadAnalyticFromCache DONE" ;
     654 }

     068 GScene* GScene::Create(Opticks* ok, GGeo* ggeo)
      69 {
      70     bool loaded = false ;
      71     GScene* scene = new GScene(ok, ggeo, loaded); // GGeo needed for m_bndlib 
      72     return scene ;
      73 }
      74 GScene* GScene::Load(Opticks* ok, GGeo* ggeo)
      75 {
      76     bool loaded = true ;
      77     GScene* scene = new GScene(ok, ggeo, loaded); // GGeo needed for m_bndlib 
      78     return scene ;
      79 }
      80 
      81 bool GScene::HasCache( Opticks* ok ) // static 
      82 {
      83     const char* idpath = ok->getIdPath();
      84     bool analytic = true ;
      85     return GGeoLib::HasCacheConstituent(idpath, analytic, 0 );
      86 }






APPROACH 
----------

* testing limited by available GDML+G4DAE export pairs

* juno processing takes too long (several minutes) for convenient test cycle, so 

  * copy opticksdata/export/DayaBay_VGDX_20140414-1300/ under a new name to act as fresh geometry test
  * OR revive G4DAE export within Opticks ? to go together with the GDML export recently revived in cfg4



Opticks::configureCheckGeometryFiles
---------------------------------------

::

     818 bool Opticks::hasGLTF() const
     819 {
     820     // lookahead to what GScene::GScene will do
     821     return NScene::Exists(getGLTFBase(), getGLTFName()) ;
     822 }
     823 
     824 
     825 void Opticks::configureCheckGeometryFiles()
     826 {
     827     if(isGLTF() && !hasGLTF())
     828     {
     829         LOG(fatal) << "gltf option is selected but there is no gltf file " ;
     830         LOG(fatal) << " GLTFBase " << getGLTFBase() ;
     831         LOG(fatal) << " GLTFName " << getGLTFName() ;
     832         LOG(fatal) << "Try to create the GLTF from GDML with eg:  op --j1707 --gdml2gltf  "  ;
     833 
     834         //setExit(true); 
     835         //assert(0);
     836     }
     837 }


TODO : relocate geocache from /usr/local/opticks/opticksdata into /usr/local/opticks/geocache
-----------------------------------------------------------------------------------------------

This long standing TODO of relocating the geocache separately from the opticksdata checkout directory, 
to avoid the very messy "hg status" in opticksdata and potential accidents, would help with 
flexibility by decoupling source geometry files from derived files.

This will mean switching "opticksdata" into "geocache" in the paths 
of all derived files, so only source files in "opticksdata" and clean "hg status".

* OpticksResource will need to distinguish source and derived


::

    simon:opticksdata blyth$ cd /usr/local/opticks
    simon:opticks blyth$ l
    total 256
    drwxr-xr-x   10 blyth  staff     340 Nov 28 11:43 opticksdata    ## this is the hg cloned dir 
    drwxr-xr-x  380 blyth  staff   12920 Nov 27 21:02 lib
    drwxr-xr-x   33 blyth  staff    1122 Nov 27 11:26 build
    drwxr-xr-x   20 blyth  staff     680 Sep 12 16:05 include
    drwxr-xr-x   20 blyth  staff     680 Sep 12 14:32 bin
    drwxr-xr-x   23 blyth  staff     782 Sep  4 18:10 gl
    drwxr-xr-x   21 blyth  staff     714 Jun 14 17:19 externals
    drwxr-xr-x    5 blyth  staff     170 Jun 14 16:23 installcache
    -rw-r--r--@   1 blyth  staff  127384 Jun 14 13:31 opticks-externals-install.txt
    simon:opticks blyth$ 

    simon:opticks blyth$ 
    simon:opticks blyth$ l opticksdata/
    total 16
    -rw-r--r--   1 blyth  staff   398 Sep 11 21:05 OpticksIDPATH.log
    drwxr-xr-x   6 blyth  staff   204 Sep 11 20:09 gensteps
    drwxr-xr-x  12 blyth  staff   408 Jul 22 10:07 export
    drwxr-xr-x   3 blyth  staff   102 Jun 14 13:13 config
    -rw-r--r--   1 blyth  staff  1150 Jun 14 13:13 opticksdata.bash
    drwxr-xr-x   3 blyth  staff   102 Jun 14 13:13 refractiveindex
    drwxr-xr-x   4 blyth  staff   136 Jun 14 13:13 resource
    simon:opticks blyth$ 




Another derived file, needing to be relocated:

::

    204 opticksdata-ini(){ echo $(opticks-prefix)/opticksdata/config/opticksdata.ini ; }
    205 opticksdata-export-ini()
    206 {
    207    local msg="=== $FUNCNAME :"
    208 
    209    opticksdata-export 
    210 
    211    local ini=$(opticksdata-ini)
    212    local dir=$(dirname $ini)
    213    mkdir -p $dir
    214 
    215    echo $msg writing OPTICKS_DAEPATH_ environment to $ini
    216    env | grep OPTICKSDATA_DAEPATH_ | sort > $ini
    217 
    218    cat $ini
    219 }


OpticksResource paths all based off the daepath
------------------------------------------------


opticksdata paths::

    simon:optickscore blyth$ cat /usr/local/opticks/opticksdata/config/opticksdata.ini
    OPTICKSDATA_DAEPATH_DFAR=/usr/local/opticks/opticksdata/export/Far_VGDX_20140414-1256/g4_00.dae
    OPTICKSDATA_DAEPATH_DLIN=/usr/local/opticks/opticksdata/export/Lingao_VGDX_20140414-1247/g4_00.dae
    OPTICKSDATA_DAEPATH_DPIB=/usr/local/opticks/opticksdata/export/dpib/cfg4.dae
    OPTICKSDATA_DAEPATH_DYB=/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae
    OPTICKSDATA_DAEPATH_J1707=/usr/local/opticks/opticksdata/export/juno1707/g4_00.dae
    OPTICKSDATA_DAEPATH_JPMT=/usr/local/opticks/opticksdata/export/juno/test3.dae
    OPTICKSDATA_DAEPATH_LXE=/usr/local/opticks/opticksdata/export/LXe/g4_00.dae
    simon:optickscore blyth$ 

geocache layout can ignore the root "/usr/local/opticks/opticksdata/export" just use ParentName::

    /usr/local/opticks/geocache/Far_VGDX_20140414-1256/
    /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/

idpath can simplify::

    /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae

    /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/
         ## this form retains the name of src file


* idfold can come from BOpticksResource
* idpath needs to be in OpticksResource as needs the digest 

::

    2017-11-28 14:08:08.203 INFO  [63474] [OpticksResource::dumpPaths@712] dumpPaths
                 daepath :  Y : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae
                gdmlpath :  Y : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml
                gltfpath :  Y : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gltf
                metapath :  N : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.ini
               g4env_ini :  Y :     /usr/local/opticks/externals/config/geant4.ini
              okdata_ini :  Y : /usr/local/opticks/opticksdata/config/opticksdata.ini
    2017-11-28 14:08:08.204 INFO  [63474] [OpticksResource::dumpDirs@741] dumpDirs
          install_prefix :  Y :                                 /usr/local/opticks
         opticksdata_dir :  Y :                     /usr/local/opticks/opticksdata
            resource_dir :  Y :            /usr/local/opticks/opticksdata/resource
                  idpath :  Y : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
              idpath_tmp :  N :                                                  -
                  idfold :  Y : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300
                  idbase :  Y :              /usr/local/opticks/opticksdata/export
           detector_base :  Y :      /usr/local/opticks/opticksdata/export/DayaBay



::


    simon:opticks blyth$ OPTICKS_RESOURCE_LAYOUT=1 BOpticksResourceTest
    2017-11-28 17:54:05.733 INFO  [158492] [BOpticksResource::Summary@367] BOpticksResource::Summary layout 1
    prefix   : /usr/local/opticks
    envprefix: OPTICKS_
    getPTXPath(generate.cu.ptx) = /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate.cu.ptx
    PTXPath(generate.cu.ptx) = /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate.cu.ptx
    debugging_idpath  /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    debugging_idfold  /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300
    usertmpdir ($TMP) /tmp/blyth/opticks
    ($TMPTEST)        /tmp/blyth/opticks/test
    2017-11-28 17:54:05.734 INFO  [158492] [BOpticksResource::dumpPaths@502] dumpPaths
                         g4env_ini :  Y :     /usr/local/opticks/externals/config/geant4.ini
                        okdata_ini :  Y : /usr/local/opticks/opticksdata/config/opticksdata.ini
                           srcpath :  Y : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae
                           daepath :  Y : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae
                          gdmlpath :  Y : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml
                          gltfpath :  Y : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gltf
                          metapath :  N : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.ini
    2017-11-28 17:54:05.735 INFO  [158492] [BOpticksResource::dumpDirs@532] dumpDirs
                    install_prefix :  Y :                                 /usr/local/opticks
                   opticksdata_dir :  Y :                     /usr/local/opticks/opticksdata
                      geocache_dir :  N :                        /usr/local/opticks/geocache
                      resource_dir :  Y :            /usr/local/opticks/opticksdata/resource
                      gensteps_dir :  Y :            /usr/local/opticks/opticksdata/gensteps
                  installcache_dir :  Y :                    /usr/local/opticks/installcache
              rng_installcache_dir :  Y :                /usr/local/opticks/installcache/RNG
              okc_installcache_dir :  Y :                /usr/local/opticks/installcache/OKC
              ptx_installcache_dir :  Y :                /usr/local/opticks/installcache/PTX
                            idfold :  N : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300
                            idpath :  N : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1
                        idpath_tmp :  N :                                                  -
    2017-11-28 17:54:05.736 INFO  [158492] [BOpticksResource::dumpNames@480] dumpNames
                            idname :  - :                         DayaBay_VGDX_20140414-1300
                            idfile :  - :                                          g4_00.dae
           OPTICKS_RESOURCE_LAYOUT :  - :                                                  1
     treedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras
    simon:opticks blyth$ 




Running with new layout before generating geocache
----------------------------------------------------

::

    87% tests passed, 36 tests failed out of 283

    Total Test time (real) = 119.24 sec

    The following tests FAILED:
        177 - GGeoTest.GMaterialLibTest (OTHER_FAULT)
        180 - GGeoTest.GScintillatorLibTest (OTHER_FAULT)
        183 - GGeoTest.GBndLibTest (OTHER_FAULT)
        184 - GGeoTest.GBndLibInitTest (OTHER_FAULT)
        195 - GGeoTest.GPartsTest (OTHER_FAULT)
        197 - GGeoTest.GPmtTest (OTHER_FAULT)
        198 - GGeoTest.BoundariesNPYTest (OTHER_FAULT)
        199 - GGeoTest.GAttrSeqTest (OTHER_FAULT)
        203 - GGeoTest.GGeoLibTest (OTHER_FAULT)
        204 - GGeoTest.GGeoTest (OTHER_FAULT)
        205 - GGeoTest.GMakerTest (OTHER_FAULT)
        212 - GGeoTest.GSurfaceLibTest (OTHER_FAULT)
        214 - GGeoTest.NLookupTest (OTHER_FAULT)
        215 - GGeoTest.RecordsNPYTest (OTHER_FAULT)
        216 - GGeoTest.GSceneTest (OTHER_FAULT)
        217 - GGeoTest.GMeshLibTest (OTHER_FAULT)
        ## got the expected errors for all the above

        222 - OpticksGeometryTest.OpticksGeometryTest (OTHER_FAULT)
        223 - OpticksGeometryTest.OpticksHubTest (OTHER_FAULT)
        ## got sensorlist errors, twas expecting 3-dot idpath structure

        241 - OptiXRapTest.OScintillatorLibTest (OTHER_FAULT)
        242 - OptiXRapTest.OOTextureTest (OTHER_FAULT)
        247 - OptiXRapTest.OOboundaryTest (OTHER_FAULT)
        248 - OptiXRapTest.OOboundaryLookupTest (OTHER_FAULT)
        252 - OptiXRapTest.OEventTest (OTHER_FAULT)
        253 - OptiXRapTest.OInterpolationTest (OTHER_FAULT)
        254 - OptiXRapTest.ORayleighTest (OTHER_FAULT)
        258 - OKOPTest.OpSeederTest (OTHER_FAULT)
        267 - cfg4Test.CMaterialLibTest (OTHER_FAULT)
        268 - cfg4Test.CMaterialTest (OTHER_FAULT)
        269 - cfg4Test.CTestDetectorTest (OTHER_FAULT)
        270 - cfg4Test.CGDMLDetectorTest (OTHER_FAULT)
        271 - cfg4Test.CGeometryTest (OTHER_FAULT)
        272 - cfg4Test.CG4Test (OTHER_FAULT)
        277 - cfg4Test.CCollectorTest (OTHER_FAULT)
        278 - cfg4Test.CInterpolationTest (OTHER_FAULT)
        280 - cfg4Test.CGROUPVELTest (OTHER_FAULT)
        283 - okg4Test.OKG4Test (OTHER_FAULT)
    Errors while running CTest
    Tue Nov 28 18:12:01 CST 2017
    opticks-t- : use -V to show output, ctest output written to /usr/local/opticks/build/ctest.log
    simon:opticks blyth$ 


Unexpected errors from 

::

    simon:opticks blyth$ OpticksGeometryTest
    2017-11-28 18:15:22.104 INFO  [180505] [Opticks::dumpArgs@968] Opticks::configure argc 1
      0 : OpticksGeometryTest
    2017-11-28 18:15:22.105 INFO  [180505] [OpticksHub::configure@236] OpticksHub::configure m_gltf 0
    2017-11-28 18:15:22.106 INFO  [180505] [OpticksHub::loadGeometry@366] OpticksHub::loadGeometry START
    2017-11-28 18:15:22.111 INFO  [180505] [NSceneConfig::NSceneConfig@50] NSceneConfig::NSceneConfig cfg [check_surf_containment=0,check_aabb_containment=0,instance_repeat_min=400,instance_vertex_min=0]
    2017-11-28 18:15:22.114 INFO  [180505] [OpticksGeometry::loadGeometry@102] OpticksGeometry::loadGeometry START 
    2017-11-28 18:15:22.114 INFO  [180505] [OpticksGeometry::loadGeometryBase@134] OpticksGeometry::loadGeometryBase START 
    2017-11-28 18:15:22.812 ERROR [180505] [NSensorList::load@88] NSensorList::load idpath is expected to be in 3-parts separted by dot eg  g4_00.gdasdyig3736781.dae  idpath 
    2017-11-28 18:15:22.812 INFO  [180505] [*OpticksResource::getSensorList@1055] OpticksResource::getSensorList NSensorList:  NSensor count 0 distinct identier count 0







::

    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/GammaYIELDRATIO.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/NeutronFASTTIMECONSTANT.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/NeutronSLOWTIMECONSTANT.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/NeutronYIELDRATIO.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/RAYLEIGH.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/REEMISSIONPROB.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/RESOLUTIONSCALE.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/RINDEX.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/ReemissionFASTTIMECONSTANT.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/ReemissionSLOWTIMECONSTANT.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/ReemissionYIELDRATIO.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/SCINTILLATIONYIELD.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/SLOWCOMPONENT.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/SLOWTIMECONSTANT.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/YIELDRATIO.npy
    ? xport/DayaBay/GSourceLib/GSourceLib.npy
    ? xport/DayaBay/GSurfaceLib/GPropertyLibMetadata.json
    ? xport/DayaBay/GSurfaceLib/GSurfaceLib.npy
    ? xport/DayaBay/GSurfaceLib/GSurfaceLibOptical.npy
    ? xport/DayaBay/MeshIndex/GItemIndexLocal.json
    ? xport/DayaBay/MeshIndex/GItemIndexSource.json
    simon:opticksgeo blyth$ 
    simon:opticksgeo blyth$ 
    simon:opticksgeo blyth$ 
    simon:opticksgeo blyth$ 
    simon:opticksgeo blyth$ OpticksGeometryTest 




Axel reports GSceneTest fail
--------------------------------

Today I got the latest updates and also did the opticks tests (opticks-t) and got the following error:

::

    99% tests passed, 1 tests failed out of 283

    Total Test time (real) = 176.07 sec

    The following tests FAILED:
        216 - GGeoTest.GSceneTest (OTHER_FAULT)
    Errors while running CTest
    Mon Nov 27 12:58:25 CET 2017


::

    gpu-CELSIUS-R940 opticks # GSceneTest 
    2017-11-27 14:33:48.056 INFO  [6897] [Opticks::dumpArgs@958] Opticks::configure argc 3
      0 : GSceneTest
      1 : --gltf
      2 : 101
    2017-11-27 14:33:48.057 FATAL [6897] [Opticks::configureCheckGeometryFiles@819] gltf option is selected but there is no gltf file 
    2017-11-27 14:33:48.057 FATAL [6897] [Opticks::configureCheckGeometryFiles@820]  GLTFBase $TMP/nd
    2017-11-27 14:33:48.058 FATAL [6897] [Opticks::configureCheckGeometryFiles@821]  GLTFName scene.gltf
    2017-11-27 14:33:48.058 FATAL [6897] [Opticks::configureCheckGeometryFiles@822] Try to create the GLTF from GDML with eg:  op --j1707 --gdml2gltf  
    2017-11-27 14:33:48.058 INFO  [6897] [main@59] GSceneTest base $TMP/nd name scene.gltf config check_surf_containment=0,check_aabb_containment=0,instance_repeat_min=400,instance_vertex_min=0 gltf 101
    2017-11-27 14:33:48.063 INFO  [6897] [NSceneConfig::NSceneConfig@50] NSceneConfig::NSceneConfig cfg [check_surf_containment=0,check_aabb_containment=0,instance_repeat_min=400,instance_vertex_min=0]
    2017-11-27 14:33:48.071 INFO  [6897] [GMaterialLib::postLoadFromCache@70] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2017-11-27 14:33:48.072 INFO  [6897] [GMaterialLib::replaceGROUPVEL@560] GMaterialLib::replaceGROUPVEL  ni 38
    2017-11-27 14:33:48.083 INFO  [6897] [GGeoLib::loadConstituents@161] GGeoLib::loadConstituents mm.reldir GMergedMesh gp.reldir GParts MAX_MERGED_MESH  10
    2017-11-27 14:33:48.083 INFO  [6897] [GGeoLib::loadConstituents@168] /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    2017-11-27 14:33:48.184 INFO  [6897] [GGeoLib::loadConstituents@217] GGeoLib::loadConstituents loaded 6 ridx (  0,  1,  2,  3,  4,  5,)
    2017-11-27 14:33:48.248 INFO  [6897] [GMeshLib::loadMeshes@219] idpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    2017-11-27 14:33:48.282 INFO  [6897] [GGeo::loadAnalyticFromCache@651] GGeo::loadAnalyticFromCache START
    2017-11-27 14:33:48.354 INFO  [6897] [OpticksResource::getSensorList@1248] OpticksResource::getSensorList NSensorList:  NSensor count 6888 distinct identier count 684
    2017-11-27 14:33:48.354 INFO  [6897] [GGeoLib::loadConstituents@161] GGeoLib::loadConstituents mm.reldir GMergedMeshAnalytic gp.reldir GPartsAnalytic MAX_MERGED_MESH  10
    2017-11-27 14:33:48.354 INFO  [6897] [GGeoLib::loadConstituents@168] /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    2017-11-27 14:33:48.354 INFO  [6897] [GGeoLib::loadConstituents@217] GGeoLib::loadConstituents loaded 0 ridx ()
    2017-11-27 14:33:48.354 WARN  [6897] [GItemList::load_@66] GItemList::load_ NO SUCH TXTPATH /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GNodeLibAnalytic/PVNames.txt
    2017-11-27 14:33:48.354 WARN  [6897] [GItemList::load_@66] GItemList::load_ NO SUCH TXTPATH /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GNodeLibAnalytic/LVNames.txt
    2017-11-27 14:33:48.354 WARN  [6897] [Index::load@420] Index::load FAILED to load index  idpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae itemtype GItemIndex Source path /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/MeshIndexAnalytic/GItemIndexSource.json Local path /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/MeshIndexAnalytic/GItemIndexLocal.json
    2017-11-27 14:33:48.354 WARN  [6897] [GItemIndex::loadIndex@176] GItemIndex::loadIndex failed for  idpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae reldir MeshIndexAnalytic override NULL
    2017-11-27 14:33:48.354 FATAL [6897] [GMeshLib::loadFromCache@61]  meshindex load failure 
    GSceneTest: /home/gpu/opticks/ggeo/GMeshLib.cc:62: void GMeshLib::loadFromCache(): Assertion `has_index && " MISSING MESH INDEX : PERHAPS YOU NEED TO CREATE/RE-CREATE GEOCACHE WITH : op.sh -G "' failed.
    Aborted

I ran "op -G", but still the error occurs.




Succeeding GSceneTest
-----------------------

* note double load of GGeoLib, seems GScene not using the basis geometry approach ?



My successful GSceneTest::

    simon:issues blyth$ GSceneTest 
    2017-11-28 12:14:52.023 INFO  [36458] [Opticks::dumpArgs@968] Opticks::configure argc 3
      0 : GSceneTest
      1 : --gltf
      2 : 101
    2017-11-28 12:14:52.024 FATAL [36458] [Opticks::configureCheckGeometryFiles@829] gltf option is selected but there is no gltf file 
    2017-11-28 12:14:52.024 FATAL [36458] [Opticks::configureCheckGeometryFiles@830]  GLTFBase $TMP/nd
    2017-11-28 12:14:52.024 FATAL [36458] [Opticks::configureCheckGeometryFiles@831]  GLTFName scene.gltf
    2017-11-28 12:14:52.024 FATAL [36458] [Opticks::configureCheckGeometryFiles@832] Try to create the GLTF from GDML with eg:  op --j1707 --gdml2gltf  
    2017-11-28 12:14:52.024 INFO  [36458] [main@62] GSceneTest base $TMP/nd name scene.gltf config check_surf_containment=0,check_aabb_containment=0,instance_repeat_min=400,instance_vertex_min=0 gltf 101
    2017-11-28 12:14:52.028 INFO  [36458] [NSceneConfig::NSceneConfig@50] NSceneConfig::NSceneConfig cfg [check_surf_containment=0,check_aabb_containment=0,instance_repeat_min=400,instance_vertex_min=0]
    2017-11-28 12:14:52.031 ERROR [36458] [GSceneTest::GSceneTest@33] loadFromCache
    2017-11-28 12:14:52.034 INFO  [36458] [GMaterialLib::postLoadFromCache@70] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2017-11-28 12:14:52.034 INFO  [36458] [GMaterialLib::replaceGROUPVEL@560] GMaterialLib::replaceGROUPVEL  ni 38
    2017-11-28 12:14:52.040 INFO  [36458] [GGeoLib::loadConstituents@161] GGeoLib::loadConstituents mm.reldir GMergedMesh gp.reldir GParts MAX_MERGED_MESH  10
    2017-11-28 12:14:52.040 INFO  [36458] [GGeoLib::loadConstituents@168] /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    2017-11-28 12:14:52.171 INFO  [36458] [GGeoLib::loadConstituents@217] GGeoLib::loadConstituents loaded 6 ridx (  0,  1,  2,  3,  4,  5,)
    2017-11-28 12:14:52.257 INFO  [36458] [GMeshLib::loadMeshes@219] idpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    2017-11-28 12:14:52.290 ERROR [36458] [GSceneTest::GSceneTest@35] loadAnalyticFromCache
    2017-11-28 12:14:52.290 INFO  [36458] [GGeo::loadAnalyticFromCache@651] GGeo::loadAnalyticFromCache START
    2017-11-28 12:14:52.456 INFO  [36458] [*OpticksResource::getSensorList@1248] OpticksResource::getSensorList NSensorList:  NSensor count 6888 distinct identier count 684
    2017-11-28 12:14:52.456 INFO  [36458] [GGeoLib::loadConstituents@161] GGeoLib::loadConstituents mm.reldir GMergedMeshAnalytic gp.reldir GPartsAnalytic MAX_MERGED_MESH  10
    2017-11-28 12:14:52.456 INFO  [36458] [GGeoLib::loadConstituents@168] /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    2017-11-28 12:14:52.603 INFO  [36458] [GGeoLib::loadConstituents@217] GGeoLib::loadConstituents loaded 6 ridx (  0,  1,  2,  3,  4,  5,)
    2017-11-28 12:14:52.679 INFO  [36458] [GMeshLib::loadMeshes@219] idpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    2017-11-28 12:14:53.220 INFO  [36458] [GGeo::loadAnalyticFromCache@653] GGeo::loadAnalyticFromCache DONE
    2017-11-28 12:14:53.220 ERROR [36458] [GSceneTest::GSceneTest@37] dumpStats
    GGeo::dumpStats
     mm  0 : vertices  204464 faces  403712 transforms   12230 itransforms       1 
     mm  1 : vertices       0 faces       0 transforms       1 itransforms    1792 
     mm  2 : vertices       8 faces      12 transforms       1 itransforms     864 
     mm  3 : vertices       8 faces      12 transforms       1 itransforms     864 
     mm  4 : vertices       8 faces      12 transforms       1 itransforms     864 
     mm  5 : vertices    1474 faces    2928 transforms       5 itransforms     672 
       totVertices    205962  totFaces    406676 
      vtotVertices   1215728 vtotFaces   2402432 (virtual: scaling by transforms)
      vfacVertices     5.903 vfacFaces     5.907 (virtual to total ratio)
    simon:issues blyth$ 


