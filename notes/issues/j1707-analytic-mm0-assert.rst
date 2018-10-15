j1707-analytic-mm0-assert
============================


FIXED By reconstructing geocache and extras
----------------------------------------------


So try going back to gdml2gltf too::

    op.sh --j1707 --gdml2gltf

    op.sh --j1707 --gltf 3 -G

    op.sh --j1707 --gltf 3 --tracer



analytic mm0 assert
------------------------

::

   op --j1707 --tracer --gltf 3


::

    2018-10-15 14:44:26.677 FATAL [25743] [OpticksHub::registerGeometry@514] OpticksHub::registerGeometry
    OTracerTest: /home/blyth/opticks/opticksgeo/OpticksHub.cc:516: void OpticksHub::registerGeometry(): Assertion `mm0' failed.
    /home/blyth/opticks/bin/op.sh: line 876: 25743 Aborted                 (core dumped) /home/blyth/local/opticks/lib/OTracerTest --size 1920,1080,1 --position 100,100 --j1707 --tracer --gltf 3
    /home/blyth/opticks/bin/op.sh RC 134


triangulated working
----------------------

::

   op --j1707 --tracer 



first thing to try : recreate geocache
------------------------------------------

::

   op.sh --j1707 --gltf 3 -G


Apppears to work normally until at the end::

    2018-10-15 14:53:17.485 INFO  [26229] [Index::save@348] Index::save sname GItemIndexSource.json lname GItemIndexLocal.json itemtype GItemIndex ext .json idpath /home/blyth/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae dir /home/blyth/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/MeshIndex
    2018-10-15 14:53:17.499 INFO  [26229] [GMeshLib::writeMeshUsage@361]  write to /home/blyth/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMeshLib/MeshUsage.txt
    2018-10-15 14:53:17.643 FATAL [26229] [GMaterialLib::save@67] [
    2018-10-15 14:53:17.644 FATAL [26229] [GMaterialLib::save@69] ]
    2018-10-15 14:53:17.644 INFO  [26229] [NMeta::write@206] write to /home/blyth/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GSurfaceLib/GPropertyLibMetadata.json
    2018-10-15 14:53:17.644 ERROR [26229] [GPropertyLib::saveToCache@466] GPropertyLib::saveToCache dir /home/blyth/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GSurfaceLib name GSurfaceLibOptical.npy type GSurfaceLib
    2018-10-15 14:53:17.647 ERROR [26229] [GPropertyLib::saveToCache@466] GPropertyLib::saveToCache dir /home/blyth/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GBndLib name GBndLibIndex.npy type GBndLib
    2018-10-15 14:53:17.648 INFO  [26229] [NMeta::write@206] write to /home/blyth/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/cachemeta.json
    2018-10-15 14:53:17.648 FATAL [26229] [GGeo::save@714] ]
    2018-10-15 14:53:17.648 INFO  [26229] [GGeo::loadAnalyticFromGLTF@670] GGeo::loadAnalyticFromGLTF START
    2018-10-15 14:53:17.648 INFO  [26229] [NGLTF::load@52] NGLTF::load path /home/blyth/local/opticks/opticksdata/export/juno1707/g4_00.gltf
    2018-10-15 14:53:23.418 INFO  [26229] [NGLTF::load@79] NGLTF::load DONE
    2018-10-15 14:53:25.093 INFO  [26229] [NScene::init_lvlists@272]  csgskip_path /home/blyth/local/opticks/opticksdata/export/juno1707/g4_00/CSGSKIP_DEEP_TREES.txt
    2018-10-15 14:53:25.093 INFO  [26229] [NScene::init_lvlists@273]  placeholder_path /home/blyth/local/opticks/opticksdata/export/juno1707/g4_00/PLACEHOLDER_FAILED_POLY.txt
    2018-10-15 14:53:25.093 INFO  [26229] [NScene::init@177] NScene::init START num_nodes 290276
    2018-10-15 14:53:25.093 INFO  [26229] [NScene::load_csg_metadata@357] NScene::load_csg_metadata verbosity 1 num_meshes 35
    2018-10-15 14:53:25.093 WARN  [26229] [NScene::getCSGMeta@425]  missing ALL metadata for mesh_id  0
    /home/blyth/opticks/bin/op.sh: line 876: 26229 Segmentation fault      (core dumped) /home/blyth/local/opticks/lib/OKTest --size 1920,1080,1 --position 100,100 --j1707 --gltf 3 -G
    /home/blyth/opticks/bin/op.sh RC 139
    [blyth@localhost ~]$ 


