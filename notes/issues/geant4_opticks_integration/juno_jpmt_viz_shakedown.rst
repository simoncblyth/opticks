JUNO JPMT Shakedown
=====================


Issue : default torch source goes off somewhere in TT
--------------------------------------------------------

TODO: reposition into CD 

Reproduce::

    op --jpmt 


Issue : segv
---------------

Segv with new override --noload::

    simon:optickscore blyth$ tviz-jpmt-cerenkov --noload
    288 -rwxr-xr-x  1 blyth  staff  145904 Dec  8 21:07 /usr/local/opticks/lib/OKTest
    proceeding : /usr/local/opticks/lib/OKTest --jpmt --jwire --target 64670 --load --animtimemax 200 --timemax 200 --optixviz --fullscreen --cerenkov --noload
    2016-12-08 21:09:08.884 INFO  [500343] [OpticksDbg::postconfigure@49] OpticksDbg::postconfigure OpticksDbg  debug_photon  size: 0 elem: () other_photon  size: 0 elem: ()
    2016-12-08 21:09:08.884 WARN  [500343] [BTree::loadTree@48] BTree.loadTree: can't find file /usr/local/opticks/opticksdata/export/juno/ChromaMaterialMap.json
    2016-12-08 21:09:08.884 FATAL [500343] [NSensorList::read@122] NSensorList::read failed to open /usr/local/opticks/opticksdata/export/juno/test3.idmap
    2016-12-08 21:09:08.884 INFO  [500343] [*GMergedMesh::load@591] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae/GMergedMesh/0 -> cachedir /usr/local/opticks/opticksdata/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae/GMergedMesh/0 index 0 version (null) existsdir 1
    2016-12-08 21:09:09.142 INFO  [500343] [*GMergedMesh::load@591] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae/GMergedMesh/1 -> cachedir /usr/local/opticks/opticksdata/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae/GMergedMesh/1 index 1 version (null) existsdir 1
    2016-12-08 21:09:09.180 INFO  [500343] [*GMergedMesh::load@591] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae/GMergedMesh/2 -> cachedir /usr/local/opticks/opticksdata/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae/GMergedMesh/2 index 2 version (null) existsdir 1
    2016-12-08 21:09:09.202 INFO  [500343] [*GMergedMesh::load@591] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae/GMergedMesh/3 -> cachedir /usr/local/opticks/opticksdata/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae/GMergedMesh/3 index 3 version (null) existsdir 1
    2016-12-08 21:09:09.204 INFO  [500343] [*GMergedMesh::load@591] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae/GMergedMesh/4 -> cachedir /usr/local/opticks/opticksdata/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae/GMergedMesh/4 index 4 version (null) existsdir 1
    2016-12-08 21:09:09.218 INFO  [500343] [GMaterialLib::postLoadFromCache@67] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2016-12-08 21:09:09.218 INFO  [500343] [GMaterialLib::replaceGROUPVEL@552] GMaterialLib::replaceGROUPVEL  ni 17
    2016-12-08 21:09:09.218 INFO  [500343] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [Acrylic]
    2016-12-08 21:09:09.218 INFO  [500343] [GPropertyLib::close@384] GPropertyLib::close type GMaterialLib buf 17,2,39,4
    2016-12-08 21:09:09.220 WARN  [500343] [*GPmt::load@51] GPmt::load resource does not exist /usr/local/opticks/opticksdata/export/juno/GPmt/0
    2016-12-08 21:09:09.228 WARN  [500343] [OpticksGen::initInputGensteps@58] OpticksGen::initInputGensteps SKIP as isNoInputGensteps 
    2016-12-08 21:09:09.228 INFO  [500343] [SLog::operator@15] OpticksHub::OpticksHub DONE
    2016-12-08 21:09:09.230 FATAL [500343] [OpticksHub::configureState@196] OpticksHub::configureState NState::description /Users/blyth/.opticks/juno/State state dir /Users/blyth/.opticks/juno/State
    2016-12-08 21:09:09.230 WARN  [500343] [OpticksViz::prepareScene@179] disable GeometryStyle  WIRE for JUNO as too slow 
    2016-12-08 21:09:09.230 INFO  [500343] [OpticksViz::prepareScene@190] App::prepareViz axis,genstep,nopstep,photon,record,bb0,bb1,
    Full screen windows cannot be moved2016-12-08 21:09:09.881 INFO  [500343] [OpticksViz::uploadGeometry@229] Opticks time 0.0000,200.0000,200.0000,0.0000 space 0.0000,0.0000,9300.0000,33550.0000 wavelength 60.0000,820.0000,20.0000,760.0000
    2016-12-08 21:09:09.998 INFO  [500343] [OpticksGeometry::setTarget@129] OpticksGeometry::setTarget  target 0 aim 1 ce  0 0 9300 33550
    2016-12-08 21:09:09.998 INFO  [500343] [Composition::setCenterExtent@991] Composition::setCenterExtent ce 0.0000,0.0000,9300.0000,33550.0000
    2016-12-08 21:09:09.998 INFO  [500343] [SLog::operator@15] OpticksViz::OpticksViz DONE
    2016-12-08 21:09:12.923 INFO  [500343] [SLog::operator@15] OScene::OScene DONE
    2016-12-08 21:09:12.923 FATAL [500343] [*OContext::addEntry@44] OContext::addEntry G
    2016-12-08 21:09:12.923 INFO  [500343] [SLog::operator@15] OEvent::OEvent DONE
    2016-12-08 21:09:14.246 INFO  [500343] [SLog::operator@15] OPropagator::OPropagator DONE
    2016-12-08 21:09:14.246 INFO  [500343] [SLog::operator@15] OpEngine::OpEngine DONE
    2016-12-08 21:09:14.265 FATAL [500343] [*OContext::addEntry@44] OContext::addEntry P
    2016-12-08 21:09:14.265 INFO  [500343] [SLog::operator@15] OKGLTracer::OKGLTracer DONE
    2016-12-08 21:09:14.265 INFO  [500343] [SLog::operator@15] OKPropagator::OKPropagator DONE
    OKMgr::init
       OptiXVersion :            3080
    2016-12-08 21:09:14.265 INFO  [500343] [SLog::operator@15] OKMgr::OKMgr DONE
    /Users/blyth/opticks/bin/op.sh: line 580: 54224 Segmentation fault: 11  /usr/local/opticks/lib/OKTest --jpmt --jwire --target 64670 --load --animtimemax 200 --timemax 200 --optixviz --fullscreen --cerenkov --noload
    /Users/blyth/opticks/bin/op.sh RC 139
    simon:optickscore blyth$ 
    simon:optickscore blyth$ 
    simon:optickscore blyth$ tviz-jpmt-cerenkov --noload


