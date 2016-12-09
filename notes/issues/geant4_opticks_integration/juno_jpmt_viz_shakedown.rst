JUNO JPMT Shakedown
=====================


Issue : default torch source goes off somewhere in TT
--------------------------------------------------------

TODO: reposition into CD 

Reproduce::

    op --jpmt 


Issue : segv due to no input gensteps
----------------------------------------

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



::

    tviz-;tviz-jpmt-cerenkov --noload --debugger


    (lldb) bt
    * thread #1: tid = 0x8265a, 0x000000010067cb8d libNPY.dylib`NPYBase::getItemShape(unsigned int) [inlined] std::__1::vector<int, std::__1::allocator<int> >::size(this=0x0000000000000070) const at vector:656, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x78)
      * frame #0: 0x000000010067cb8d libNPY.dylib`NPYBase::getItemShape(unsigned int) [inlined] std::__1::vector<int, std::__1::allocator<int> >::size(this=0x0000000000000070) const at vector:656
        frame #1: 0x000000010067cb8d libNPY.dylib`NPYBase::getItemShape(this=0x0000000000000000, ifr=0) + 1341 at NPYBase.cpp:592
        frame #2: 0x0000000100679c62 libNPY.dylib`NPYBase::getShapeString(this=0x0000000000000000, ifr=0) + 34 at NPYBase.cpp:586
        frame #3: 0x000000010097d5f4 libOpticksCore.dylib`OpticksRun::setGensteps(this=0x00000001055220f0, gensteps=0x0000000000000000) + 180 at OpticksRun.cc:77
        frame #4: 0x00000001037be8b0 libOK.dylib`OKMgr::propagate(this=0x00007fff5fbfea78) + 272 at OKMgr.cc:94
        frame #5: 0x000000010000a9f2 OKTest`main(argc=15, argv=0x00007fff5fbfeb50) + 1378 at OKTest.cc:61
        frame #6: 0x00007fff8aded5fd libdyld.dylib`start + 1
    (lldb) 

    (lldb) f 4
    frame #4: 0x00000001037be8b0 libOK.dylib`OKMgr::propagate(this=0x00007fff5fbfea98) + 272 at OKMgr.cc:94
       91           {
       92               m_run->createEvent(i);
       93   
    -> 94               m_run->setGensteps(m_gen->getInputGensteps()); 
       95   
       96               m_propagator->propagate();
       97   

       75   void OpticksRun::setGensteps(NPY<float>* gensteps)
       76   {
    -> 77       LOG(info) << "OpticksRun::setGensteps " << gensteps->getShapeString() ;  
       78   
       79       assert(m_evt && m_g4evt && "must OpticksRun::createEvent prior to OpticksRun::setGensteps");
       80   



::

    2016-12-09 12:33:57.668 INFO  [538973] [SLog::operator@15] OKMgr::OKMgr DONE
    2016-12-09 12:33:57.668 FATAL [538973] [OpticksRun::setGensteps@78] OpticksRun::setGensteps given NULL gensteps
    Assertion failed: (!no_gensteps), function setGensteps, file /Users/blyth/opticks/optickscore/OpticksRun.cc, line 79.
    Process 62799 stopped


Hmm the tviz was setup prior to OKG4 so --noload is not going to work without some effort.





