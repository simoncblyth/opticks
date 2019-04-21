direct_geocache_rerun_showing_wrong_geometry
===============================================


RESOLVED : need to set an appropriate target on rerunning
------------------------------------------------------------

::

    OKTest --envkey --xanalytic --target 10000


Questions
-------------

* Why is the default viewpoint from the creation run not being used by the rerun ?


Observations of issue
------------------------

Creating a geocache like below succeeds to visualize
and populate geocache dir (after moving aside prior to avoid possible issue
of geocache co-mingling from digest failing to update)::

    opticksdata- 
    OKX4Test --gdmlpath $(opticksdata-jv3) --csgskiplv 22  

Reporting at cleanup::

    2019-04-19 21:58:45.485 INFO  [342222] [Opticks::cleanup@2277] Opticks.desc
                 BOpticksKey  : KEYSOURCE
          spec (OPTICKS_KEY)  : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
                     exename  : OKX4Test
             current_exename  : OKX4Test
                       class  : X4PhysicalVolume
                     volname  : lWorld0x4bc2710_PV
                      digest  : 528f4cefdac670fffe846377973af10a
                      idname  : OKX4Test_lWorld0x4bc2710_PV_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    IdPath : /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe846377973af10a/1


A subsequent run looks normal but the visualization is of a simple box::

    blyth@localhost okg4]$ env | grep OPTICKS_KEY
    OPTICKS_KEY=OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
    [blyth@localhost okg4]$ OKTest --envkey --xanalytic
    2019-04-19 22:01:35.927 INFO  [351233] [BOpticksKey::SetKey@45] BOpticksKey::SetKey from OPTICKS_KEY envvar OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
    2019-04-19 22:01:35.932 WARN  [351233] [BTree::loadTree@49] BTree.loadTree: can't find file /home/blyth/local/opticks/opticksdata/export/OKX4Test/ChromaMaterialMap.json
    2019-04-19 22:01:35.932 INFO  [351233] [OpticksHub::loadGeometry@475] OpticksHub::loadGeometry START
    2019-04-19 22:01:35.932 INFO  [351233] [OpticksGeometry::loadGeometry@87] OpticksGeometry::loadGeometry START 
    2019-04-19 22:01:35.932 ERROR [351233] [OpticksGeometry::loadGeometryBase@119] OpticksGeometry::loadGeometryBase START 
    2019-04-19 22:01:35.932 INFO  [351233] [GGeo::loadGeometry@569] GGeo::loadGeometry START loaded 1 gltf 0
    2019-04-19 22:01:35.932 ERROR [351233] [GGeo::loadFromCache@771] GGeo::loadFromCache START
    2019-04-19 22:01:35.934 INFO  [351233] [GMaterialLib::postLoadFromCache@99] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0
    2019-04-19 22:01:35.937 INFO  [351233] [GGeoLib::loadConstituents@175] GGeoLib::loadConstituents mm.reldir GMergedMesh gp.reldir GParts MAX_MERGED_MESH  10
    2019-04-19 22:01:35.937 INFO  [351233] [GGeoLib::loadConstituents@182] /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe846377973af10a/1
    2019-04-19 22:01:36.043 INFO  [351233] [GGeoLib::loadConstituents@231] GGeoLib::loadConstituents loaded 6 ridx (  0,  1,  2,  3,  4,  5,)
    2019-04-19 22:01:36.286 INFO  [351233] [GMeshLib::loadMeshes@223] idpath /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe846377973af10a/1
    2019-04-19 22:01:36.296 ERROR [351233] [GGeo::loadFromCache@792] GGeo::loadFromCache DONE
    2019-04-19 22:01:36.315 INFO  [351233] [GGeo::loadGeometry@612] GGeo::loadGeometry DONE


