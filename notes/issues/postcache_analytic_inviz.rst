postcache_analytic_inviz
============================


Issue 
----------

::

    op --gltf 1 --tracer --debugger

Postcache analytic geometry appears to load, 
but nothing visible in tracer.



::

    2017-08-29 20:56:29.779 INFO  [1665179] [SLog::operator@15] OpticksViz::OpticksViz DONE
    2017-08-29 20:56:29.928 INFO  [1665179] [OScene::init@108] OScene::init (OContext) stack_size_bytes: 2180
    2017-08-29 20:56:29.928 INFO  [1665179] [OFunc::convert@28] OFunc::convert ptxname solve_callable.cu.ptx ctxname solve_callable funcnames  SolveCubicCallable num_funcs 1
    2017-08-29 20:56:29.946 INFO  [1665179] [OFunc::convert@44] OFunc::convert id 1 name SolveCubicCallable
    2017-08-29 20:56:29.946 INFO  [1665179] [*OpticksHub::getGGeoBase@340] OpticksHub::getGGeoBase analytic switch   m_gltf 1 ggb GScene
    2017-08-29 20:56:29.946 INFO  [1665179] [OScene::init@122] OScene::init ggeobase identifier : GScene
    2017-08-29 20:56:29.946 INFO  [1665179] [OGeo::convert@169] OGeo::convert START  numMergedMesh: 6
    2017-08-29 20:56:29.946 INFO  [1665179] [GGeoLib::dump@247] OGeo::convert GGeoLib
    2017-08-29 20:56:29.946 INFO  [1665179] [GGeoLib::dump@248] GGeoLib ANALYTIC  numMergedMesh 6
    mm i   0 geocode   A                  numSolids      12230 numFaces      403712 numITransforms           1
    mm i   1 geocode   A            EMPTY numSolids          1 numFaces           0 numITransforms        1792
    mm i   2 geocode   A                  numSolids          1 numFaces          12 numITransforms         864
    mm i   3 geocode   A                  numSolids          1 numFaces          12 numITransforms         864
    mm i   4 geocode   A                  numSolids          1 numFaces          12 numITransforms         864
    mm i   5 geocode   A                  numSolids          5 numFaces        2928 numITransforms         672
    2017-08-29 20:56:29.946 WARN  [1665179] [OGeo::makeAnalyticGeometry@477] OGeo::makeAnalyticGeometry START verbosity 0 mm 0
    2017-08-29 20:56:29.946 INFO  [1665179] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GSurfaceLib TRIGGERED A CLOSE  shortname []
    2017-08-29 20:56:29.947 INFO  [1665179] [GPropertyLib::close@384] GPropertyLib::close type GSurfaceLib buf 48,2,39,4
    2017-08-29 20:56:30.250 WARN  [1665179] [OGeo::convertMergedMesh@222] OGeo::convertMesh skipping mesh 1
    2017-08-29 20:56:30.250 WARN  [1665179] [OGeo::makeAnalyticGeometry@477] OGeo::makeAnalyticGeometry START verbosity 0 mm 2
    2017-08-29 20:56:30.321 WARN  [1665179] [OGeo::makeAnalyticGeometry@477] OGeo::makeAnalyticGeometry START verbosity 0 mm 3
    2017-08-29 20:56:30.336 WARN  [1665179] [OGeo::makeAnalyticGeometry@477] OGeo::makeAnalyticGeometry START verbosity 0 mm 4
    2017-08-29 20:56:30.351 WARN  [1665179] [OGeo::makeAnalyticGeometry@477] OGeo::makeAnalyticGeometry START verbosity 0 mm 5
    2017-08-29 20:56:30.369 INFO  [1665179] [SLog::operator@15] OScene::OScene DONE
    2017-08-29 20:56:30.369 WARN  [1665179] [OpEngine::init@65] OpEngine::init skip initPropagation as tracer mode is active  
    2017-08-29 20:56:30.369 INFO  [1665179] [SLog::operator@15] OpEngine::OpEngine DONE
    2017-08-29 20:56:30.388 FATAL [1665179] [*OContext::addEntry@44] OContext::addEntry P
    2017-08-29 20:56:30.388 INFO  [1665179] [SLog::operator@15] OKGLTracer::OKGLTracer DONE
    2017-08-29 20:56:30.388 INFO  [1665179] [SLog::operator@15] OKPropagator::OKPropagator DONE
    OKMgr::init
       OptiXVersion :            3080
    2017-08-29 20:56:30.388 INFO  [1665179] [SLog::operator@15] OKMgr::OKMgr DONE
    2017-08-29 20:56:30.388 INFO  [1665179] [Bookmarks::create@249] Bookmarks::create : persisting state to slot 0
    2017-08-29 20:56:30.388 INFO  [1665179] [Bookmarks::collect@273] Bookmarks::collect 0
    2017-08-29 20:56:30.391 WARN  [1665179] [OpticksViz::prepareGUI@366] App::prepareGUI NULL TimesTable 
    2017-08-29 20:56:30.391 INFO  [1665179] [OpticksViz::renderLoop@447] enter runloop 
    2017-08-29 20:56:30.436 INFO  [1665179] [OpticksViz::renderLoop@452] after frame.show() 
    2017-08-29 20:56:30.522 INFO  [1665179] [Animator::Summary@313] Composition::gui setup Animation   OFF 0/0/    0.0000



Note that usual bounds dumping from GPU doesnt show up, eg with tboolean-torus::

    2017-08-29 21:00:05.539 INFO  [1666698] [SLog::operator@15] OKMgr::OKMgr DONE
    2017-08-29 21:00:05.540 INFO  [1666698] [OpticksRun::setGensteps@81] OpticksRun::setGensteps 1,6,4
    2017-08-29 21:00:05.540 INFO  [1666698] [OpticksRun::passBaton@95] OpticksRun::passBaton nopstep 0x7fbfe5b40d50 genstep 0x7fbfe0698c70
    2017-08-29 21:00:05.540 FATAL [1666698] [OKPropagator::propagate@65] OKPropagator::propagate(1) OK INTEROP DEVELOPMENT
    2017-08-29 21:00:05.540 INFO  [1666698] [Composition::setCenterExtent@991] Composition::setCenterExtent ce 0.0000,0.0000,0.0000,400.0000
    2017-08-29 21:00:05.540 INFO  [1666698] [OpticksHub::target@505] OpticksHub::target (geocenter) mmce 0.0000,0.0000,0.0000,400.0000
    2017-08-29 21:00:05.540 INFO  [1666698] [OpticksViz::uploadEvent@289] OpticksViz::uploadEvent (1)
    2017-08-29 21:00:05.542 INFO  [1666698] [Rdr::upload@303]       axis_attr vpos cn        3 sh                3,3,4 id    12 dt   0x7fbfe0600970 hd     Y nb        144 GL_STATIC_DRAW
    2017-08-29 21:00:05.543 INFO  [1666698] [Rdr::upload@303]    genstep_attr vpos cn        1 sh                1,6,4 id    13 dt   0x7fbfe0698ad0 hd     Y nb         96 GL_STATIC_DRAW
    2017-08-29 21:00:05.547 INFO  [1666698] [Rdr::upload@303]    nopstep_attr vpos cn        0 sh                0,4,4 id    14 dt              0x0 hd     N nb          0 GL_STATIC_DRAW
    2017-08-29 21:00:05.548 INFO  [1666698] [Rdr::upload@303]     photon_attr vpos cn    10000 sh            10000,4,4 id    15 dt              0x0 hd     N nb     640000 GL_DYNAMIC_DRAW
    2017-08-29 21:00:05.561 INFO  [1666698] [Rdr::upload@303]     record_attr rpos cn   100000 sh         10000,10,2,4 id    16 dt              0x0 hd     N nb    1600000 GL_STATIC_DRAW
    2017-08-29 21:00:05.584 INFO  [1666698] [Rdr::upload@303]   sequence_attr phis cn    10000 sh            10000,1,2 id    17 dt              0x0 hd     N nb     160000 GL_STATIC_DRAW
    2017-08-29 21:00:05.584 INFO  [1666698] [Rdr::upload@303]     phosel_attr psel cn    10000 sh            10000,1,4 id    18 dt              0x0 hd     N nb      40000 GL_STATIC_DRAW
    2017-08-29 21:00:05.584 INFO  [1666698] [Rdr::upload@303]     recsel_attr rsel cn   100000 sh         10000,10,1,4 id    19 dt              0x0 hd     N nb     400000 GL_STATIC_DRAW
    2017-08-29 21:00:05.584 INFO  [1666698] [OpticksViz::uploadEvent@296] OpticksViz::uploadEvent (1) DONE 
    2017-08-29 21:00:05.584 INFO  [1666698] [OEvent::createBuffers@62] OEvent::createBuffers  genstep 1,6,4 nopstep 0,4,4 photon 10000,4,4 record 10000,10,2,4 phosel 10000,1,4 recsel 10000,10,1,4 sequence 10000,1,2 seed 10000,1,1 hit 0,4,4
    2017-08-29 21:00:05.585 INFO  [1666698] [OEvent::uploadGensteps@242] OEvent::uploadGensteps (INTEROP) SKIP OpenGL BufferId 13
    2017-08-29 21:00:05.585 INFO  [1666698] [OpSeeder::seedComputeSeedsFromInteropGensteps@64] OpSeeder::seedComputeSeedsFromInteropGensteps : WITH_SEED_BUFFER 
    2017-08-29 21:00:05.593 INFO  [1666698] [OContext::close@219] OContext::close numEntryPoint 2
    ## intersect_analytic.cu:bounds pts:   2 pln:   0 trs:   6 
    ##csg_bounds_prim primIdx   0 partOffset   0 numParts   1 height  0 numNodes  1 tranBuffer_size   6 
    ##csg_bounds_prim primIdx   1 partOffset   1 numParts   1 height  0 numNodes  1 tranBuffer_size   6 
    ##csg_bounds_prim primIdx   0 nodeIdx  1 depth  0 elev  0 typecode 23 tranOffset  0 gtransformIdx  1 complement 0 
    ##csg_bounds_prim primIdx   1 nodeIdx  1 depth  0 elev  0 typecode  6 tranOffset  1 gtransformIdx  1 complement 0 

       1.000    0.000    0.000    0.000   (trIdx:  0)[vt]
       0.000    1.000    0.000    0.000

       1.000    0.000    0.000    0.000   (trIdx:  3)[vt]
       0.000    1.000    0.000    0.000

       0.000    0.000    1.000    0.000   (trIdx:  0)[vt]
       0.000    0.000    0.000    1.000

       0.000    0.000    1.000    0.000   (trIdx:  3)[vt]
       0.000    0.000    0.000    1.000
    // csg_bounds_torus rmajor 100.000000 rminor 50.000000 rsum 150.000000  tr 1  
    // intersect_analytic.cu:bounds primIdx 0 primFlag 101 min  -150.0000  -150.0000   -50.0000 max   150.0000   150.0000    50.0000 
    // intersect_analytic.cu:bounds primIdx 1 primFlag 101 min  -400.0000  -400.0000  -400.0000 max   400.0000   400.0000   400.0000 
    2017-08-29 21:00:06.833 INFO  [1666698] [OPropagator::prelaunch@149] 1 : (0;10000,1) prelaunch_times vali,comp,prel,lnch  0.0000 0.5444 0.5704 0.0000
    2017-08-29 21:00:06.845 INFO  [1666698] [OPropagator::launch@169] 1 : (0;10000,1) launch_times vali,comp,prel,lnch  0.0000 0.0000 0.0000 0.0116
    2017-08-29 21:00:06.845 INFO  [1666698] [OpIndexer::indexSequenceInterop@258] OpIndexer::indexSequenceInterop slicing (OBufBase*)m_seq 
    2017-08-29 21:00:06.856 INFO  [1666698] [OpticksViz::indexPresentationPrep@323] OpticksViz::indexPresentationPrep
    2017-08-29 21:00:06.859 INFO  [1666698] [GPropertyLib::close@384] GPropertyLib::close type GBndLib buf 125,4,2,39,4



