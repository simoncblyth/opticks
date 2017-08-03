j1707.all insufficient GPU memory 
====================================

NEXT
-----

* make some memory accounting tables 
* see if everything that can be instanced is being instanced
* see if anything can be dropped (Virtual things ?)

* rearrange geocache to allow changing geoselection without geocache recreation ?
  ie tie geocache lifecycle to the GDML/DAE contaning all the cached 
  ingredients that are then merged/instanced together at runtime



op --j1707 : fails to make OptiX context for lack of GPU memory 
--------------------------------------------------------------------


::

    simon:bin blyth$ op --j1707 
    ubin /usr/local/opticks/lib/OKTest cfm cmdline --j1707
    === op-export : OPTICKS_BINARY /usr/local/opticks/lib/OKTest
    288 -rwxr-xr-x  1 blyth  staff  143804 Aug  2 19:16 /usr/local/opticks/lib/OKTest
    proceeding.. : /usr/local/opticks/lib/OKTest --j1707
    2017-08-03 18:43:26.896 INFO  [1902539] [OpticksQuery::dump@79] OpticksQuery::init queryType undefined query_string all query_name NULL query_index 0 query_depth 0 no_selection 1
    2017-08-03 18:43:26.897 INFO  [1902539] [Opticks::init@319] Opticks::init DONE OpticksResource::desc digest a181a603769c1f98ad927e7367c7aa51 age.tot_seconds   1021 age.tot_minutes 17.017 age.tot_hours  0.284 age.tot_days      0.012
    2017-08-03 18:43:26.897 WARN  [1902539] [BTree::loadTree@48] BTree.loadTree: can't find file /usr/local/opticks/opticksdata/export/juno/ChromaMaterialMap.json
    2017-08-03 18:43:26.897 FATAL [1902539] [NSensorList::read@133] NSensorList::read failed to open /usr/local/opticks/opticksdata/export/juno1707/g4_00.idmap
    2017-08-03 18:43:26.898 INFO  [1902539] [*GMergedMesh::load@634] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/0 -> cachedir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/0 index 0 version (null) existsdir 1
    2017-08-03 18:43:27.160 INFO  [1902539] [*GMergedMesh::load@634] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/1 -> cachedir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/1 index 1 version (null) existsdir 1
    2017-08-03 18:43:27.203 INFO  [1902539] [*GMergedMesh::load@634] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/2 -> cachedir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/2 index 2 version (null) existsdir 1
    2017-08-03 18:43:27.229 INFO  [1902539] [*GMergedMesh::load@634] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/3 -> cachedir /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae/GMergedMesh/3 index 3 version (null) existsdir 1
    2017-08-03 18:43:27.807 INFO  [1902539] [GMeshLib::loadMeshes@206] idpath /usr/local/opticks/opticksdata/export/juno1707/g4_00.a181a603769c1f98ad927e7367c7aa51.dae
    2017-08-03 18:43:27.818 INFO  [1902539] [GMaterialLib::postLoadFromCache@67] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2017-08-03 18:43:27.818 INFO  [1902539] [GMaterialLib::replaceGROUPVEL@552] GMaterialLib::replaceGROUPVEL  ni 15
    2017-08-03 18:43:27.818 INFO  [1902539] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [Acrylic]
    2017-08-03 18:43:27.818 INFO  [1902539] [GPropertyLib::close@384] GPropertyLib::close type GMaterialLib buf 15,2,39,4
    2017-08-03 18:43:27.820 WARN  [1902539] [*GPmt::load@44] GPmt::load resource does not exist /usr/local/opticks/opticksdata/export/juno/GPmt/0
    2017-08-03 18:43:27.820 INFO  [1902539] [GGeo::loadAnalyticPmt@761] GGeo::loadAnalyticPmt AnalyticPMTIndex 0 AnalyticPMTSlice ALL Path -
    2017-08-03 18:43:27.829 INFO  [1902539] [*Opticks::makeSimpleTorchStep@1246] Opticks::makeSimpleTorchStep config  cfg NULL
    2017-08-03 18:43:27.829 INFO  [1902539] [OpticksGen::targetGenstep@130] OpticksGen::targetGenstep setting frame 3153 -0.6931,0.6589,0.2923,0.0000 0.6890,0.7248,0.0000,0.0000 -0.2119,0.2014,-0.9563,0.0000 4131.5200,-3927.3000,18648.1992,1.0000
    2017-08-03 18:43:27.829 FATAL [1902539] [GenstepNPY::setPolarization@221] GenstepNPY::setPolarization pol 0.0000,0.0000,0.0000,0.0000 npol nan,nan,nan,nan m_polw nan,nan,nan,430.0000
    2017-08-03 18:43:27.829 INFO  [1902539] [SLog::operator@15] OpticksHub::OpticksHub DONE
    2017-08-03 18:43:27.831 FATAL [1902539] [OpticksHub::configureState@200] OpticksHub::configureState NState::description /Users/blyth/.opticks/juno/State state dir /Users/blyth/.opticks/juno/State
    2017-08-03 18:43:27.831 WARN  [1902539] [OpticksViz::prepareScene@176] OpticksViz::prepareScene using non-standard rendermode 
    2017-08-03 18:43:28.841 INFO  [1902539] [OpticksViz::uploadGeometry@231] Opticks time 0.0000,200.0000,50.0000,0.0000 space 0.0000,0.0000,0.0000,60000.0000 wavelength 60.0000,820.0000,20.0000,760.0000
    2017-08-03 18:43:28.922 INFO  [1902539] [OpticksGeometry::setTarget@130] OpticksGeometry::setTarget  based on CenterExtent from m_mesh0  target 0 aim 1 ce  0 0 0 60000
    2017-08-03 18:43:28.922 INFO  [1902539] [Composition::setCenterExtent@991] Composition::setCenterExtent ce 0.0000,0.0000,0.0000,60000.0000
    2017-08-03 18:43:28.922 INFO  [1902539] [SLog::operator@15] OpticksViz::OpticksViz DONE
    2017-08-03 18:43:32.902 INFO  [1902539] [SLog::operator@15] OScene::OScene DONE
    2017-08-03 18:43:32.902 FATAL [1902539] [*OContext::addEntry@44] OContext::addEntry G
    2017-08-03 18:43:32.902 INFO  [1902539] [SLog::operator@15] OEvent::OEvent DONE
    2017-08-03 18:43:34.201 INFO  [1902539] [SLog::operator@15] OPropagator::OPropagator DONE
    2017-08-03 18:43:34.201 INFO  [1902539] [SLog::operator@15] OpEngine::OpEngine DONE
    2017-08-03 18:43:34.219 FATAL [1902539] [*OContext::addEntry@44] OContext::addEntry P
    2017-08-03 18:43:34.219 INFO  [1902539] [SLog::operator@15] OKGLTracer::OKGLTracer DONE
    2017-08-03 18:43:34.219 INFO  [1902539] [SLog::operator@15] OKPropagator::OKPropagator DONE
    OKMgr::init
       OptiXVersion :            3080
    2017-08-03 18:43:34.219 INFO  [1902539] [SLog::operator@15] OKMgr::OKMgr DONE
    2017-08-03 18:43:34.220 INFO  [1902539] [OpticksRun::setGensteps@81] OpticksRun::setGensteps 1,6,4
    2017-08-03 18:43:34.220 INFO  [1902539] [OpticksRun::passBaton@95] OpticksRun::passBaton nopstep 0x7fa62b3267a0 genstep 0x7fa62ac726a0
    2017-08-03 18:43:34.220 FATAL [1902539] [OKPropagator::propagate@65] OKPropagator::propagate(1) OK INTEROP DEVELOPMENT
    2017-08-03 18:43:34.220 INFO  [1902539] [Composition::setCenterExtent@991] Composition::setCenterExtent ce 4131.5200,-3927.3000,18648.1992,1000.0000
    2017-08-03 18:43:34.220 INFO  [1902539] [OpticksHub::target@497] OpticksHub::target evt Evt /tmp/blyth/opticks/evt/juno/torch/1 20170803_184334 /usr/local/opticks/lib/OKTest gsce 4131.5200,-3927.3000,18648.1992,1000.0000
    2017-08-03 18:43:34.220 INFO  [1902539] [OpticksViz::uploadEvent@269] OpticksViz::uploadEvent (1)
    2017-08-03 18:43:34.222 INFO  [1902539] [Rdr::upload@303]       axis_attr vpos cn        3 sh                3,3,4 id    39 dt   0x7fa628d0dd70 hd     Y nb        144 GL_STATIC_DRAW
    2017-08-03 18:43:34.223 INFO  [1902539] [Rdr::upload@303]    genstep_attr vpos cn        1 sh                1,6,4 id    40 dt   0x7fa62ac72160 hd     Y nb         96 GL_STATIC_DRAW
    2017-08-03 18:43:34.226 INFO  [1902539] [Rdr::upload@303]    nopstep_attr vpos cn        0 sh                0,4,4 id    41 dt              0x0 hd     N nb          0 GL_STATIC_DRAW
    2017-08-03 18:43:34.227 INFO  [1902539] [Rdr::upload@303]     photon_attr vpos cn   100000 sh           100000,4,4 id    42 dt              0x0 hd     N nb    6400000 GL_DYNAMIC_DRAW
    2017-08-03 18:43:34.239 INFO  [1902539] [Rdr::upload@303]     record_attr rpos cn  1000000 sh        100000,10,2,4 id    43 dt              0x0 hd     N nb   16000000 GL_STATIC_DRAW
    2017-08-03 18:43:34.260 INFO  [1902539] [Rdr::upload@303]   sequence_attr phis cn   100000 sh           100000,1,2 id    44 dt              0x0 hd     N nb    1600000 GL_STATIC_DRAW
    2017-08-03 18:43:34.261 INFO  [1902539] [Rdr::upload@303]     phosel_attr psel cn   100000 sh           100000,1,4 id    45 dt              0x0 hd     N nb     400000 GL_STATIC_DRAW
    2017-08-03 18:43:34.261 INFO  [1902539] [Rdr::upload@303]     recsel_attr rsel cn  1000000 sh        100000,10,1,4 id    46 dt              0x0 hd     N nb    4000000 GL_STATIC_DRAW
    2017-08-03 18:43:34.261 INFO  [1902539] [OpticksViz::uploadEvent@276] OpticksViz::uploadEvent (1) DONE 
    2017-08-03 18:43:34.261 INFO  [1902539] [OEvent::createBuffers@62] OEvent::createBuffers  genstep 1,6,4 nopstep 0,4,4 photon 100000,4,4 record 100000,10,2,4 phosel 100000,1,4 recsel 100000,10,1,4 sequence 100000,1,2 seed 100000,1,1 hit 0,4,4
    2017-08-03 18:43:34.261 INFO  [1902539] [OEvent::uploadGensteps@242] OEvent::uploadGensteps (INTEROP) SKIP OpenGL BufferId 40
    2017-08-03 18:43:34.261 INFO  [1902539] [OpSeeder::seedComputeSeedsFromInteropGensteps@64] OpSeeder::seedComputeSeedsFromInteropGensteps : WITH_SEED_BUFFER 
    2017-08-03 18:43:34.272 INFO  [1902539] [OContext::close@219] OContext::close numEntryPoint 2
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Unknown error (Details: Function "RTresult _rtContextCompile(RTcontext)" caught exception: Insufficient device memory. GPU does not support paging., [16515528])
    /Users/blyth/opticks/bin/op.sh: line 689: 86085 Abort trap: 6           /usr/local/opticks/lib/OKTest --j1707
    /Users/blyth/opticks/bin/op.sh RC 134




op --j1707 --tracer : Insufficient memory for full j1707 raytrace
------------------------------------------------------------------------

Full j1707 opengl viz runs, but switching to raytrace dies with GPU memory error::


    op --j1707 --tracer
    ...

    2017-08-03 18:50:33.635 INFO  [1904794] [Interactor::key_pressed@461] Interactor::key_pressed O nextRenderStyle 
    2017-08-03 18:50:33.636 INFO  [1904794] [OTracer::trace_@128] OTracer::trace  entry_index 0 trace_count 0 resolution_scale 1 size(2880,1704) ZProj.zw (-1.00348,-1700.01) front 0.5823,0.6441,-0.4960
    2017-08-03 18:50:33.636 INFO  [1904794] [OContext::close@219] OContext::close numEntryPoint 1
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Unknown error (Details: Function "RTresult _rtContextCompile(RTcontext)" caught exception: Insufficient device memory. GPU does not support paging., [16515528])
    /Users/blyth/opticks/bin/op.sh: line 689: 86511 Abort trap: 6           /usr/local/opticks/lib/OTracerTest --j1707 --tracer
    /Users/blyth/opticks/bin/op.sh RC 134
    simon:bin blyth$ 


