FIXED : Boost Update Breaks of 52/300 opticks-t tests
========================================================

Following a kitchensink macports install of gnupg21 
that included boost and some graphical libs (see w-;gpg- ~/macports/gnupg21.log )
get 52/300 opticks-t SEGFAULTs related to libboost_program_options.

FIX
-----

* recompiling okc-- gets okc-t tests to pass
* recompiling opticks-- gets all opticks-t tests to pass

Backtrace
-----------

Sampling 2 of the 52 SEGFAULT find the same cause::

    delta:issues blyth$ lldb OpticksCfgTest 
    (lldb) target create "OpticksCfgTest"
    Current executable set to 'OpticksCfgTest' (x86_64).
    (lldb) r
    (lldb) bt
    * thread #1: tid = 0x2a1e9b, 0x00007fff8aab226c libc++.1.dylib`std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::operator=(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&) + 14, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x0)
      * frame #0: 0x00007fff8aab226c libc++.1.dylib`std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >::operator=(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&) + 14
        frame #1: 0x0000000100e6fd18 libOpticksCore.dylib`boost::program_options::typed_value<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >, char>::notify(this=0x0000000103113120, value_store=0x00007fff5fbfd1f0) const + 72 at value_semantic.hpp:39
        frame #2: 0x00000001000426dd libboost_program_options-mt.dylib`boost::program_options::store(boost::program_options::basic_parsed_options<char> const&, boost::program_options::variables_map&, bool) + 1421
        frame #3: 0x000000010096340d libBoostRap.dylib`BCfg::parse_commandline(this=0x0000000104005400, argc=1, argv=0x00007fff5fbfeb30, verbose=false) + 509 at BCfg.cc:240
        frame #4: 0x000000010096307b libBoostRap.dylib`BCfg::commandline(this=0x0000000104005400, argc=1, argv=0x00007fff5fbfeb30) + 2635 at BCfg.cc:153
        frame #5: 0x000000010096319f libBoostRap.dylib`BCfg::commandline(this=0x0000000103112300, argc=1, argv=0x00007fff5fbfeb30) + 2927 at BCfg.cc:161
        frame #6: 0x0000000100003e8f OpticksCfgTest`main(argc=1, argv=0x00007fff5fbfeb30) + 911 at OpticksCfgTest.cc:23
        frame #7: 0x00007fff8c89b5fd libdyld.dylib`start + 1
    (lldb) 


Full list of fails
---------------------

::

    83% tests passed, 52 tests failed out of 300

    Total Test time (real) =  45.42 sec

    The following tests FAILED:
        160 - OpticksCoreTest.OpticksCfgTest (SEGFAULT)
        161 - OpticksCoreTest.OpticksTest (SEGFAULT)
        167 - OpticksCoreTest.OpticksAnaTest (SEGFAULT)
        168 - OpticksCoreTest.OpticksDbgTest (SEGFAULT)
        171 - OpticksCoreTest.EvtLoadTest (SEGFAULT)
        172 - OpticksCoreTest.OpticksEventAnaTest (SEGFAULT)
        173 - OpticksCoreTest.OpticksEventCompareTest (SEGFAULT)
        174 - OpticksCoreTest.OpticksEventDumpTest (SEGFAULT)
        184 - GGeoTest.GMaterialLibTest (SEGFAULT)
        191 - GGeoTest.GBndLibInitTest (SEGFAULT)
        204 - GGeoTest.GPmtTest (SEGFAULT)
        206 - GGeoTest.GAttrSeqTest (SEGFAULT)
        212 - GGeoTest.GMakerTest (SEGFAULT)
        222 - GGeoTest.RecordsNPYTest (SEGFAULT)
        223 - GGeoTest.GSceneTest (SEGFAULT)
        224 - GGeoTest.GMeshLibTest (SEGFAULT)
        225 - AssimpRapTest.AssimpRapTest (SEGFAULT)
        226 - AssimpRapTest.AssimpImporterTest (SEGFAULT)
        227 - AssimpRapTest.AssimpGGeoTest (SEGFAULT)
        229 - OpticksGeometryTest.OpticksGeometryTest (SEGFAULT)
        230 - OpticksGeometryTest.OpticksHubTest (SEGFAULT)
        231 - OpticksGeometryTest.OpenMeshRapTest (SEGFAULT)
        255 - OptiXRapTest.OOTextureTest (SEGFAULT)
        258 - OptiXRapTest.OOContextTest (SEGFAULT)
        259 - OptiXRapTest.OOContextUploadDownloadTest (SEGFAULT)
        260 - OptiXRapTest.LTOOContextUploadDownloadTest (SEGFAULT)
        261 - OptiXRapTest.OOboundaryTest (SEGFAULT)
        262 - OptiXRapTest.OOboundaryLookupTest (SEGFAULT)
        265 - OptiXRapTest.bufferTest (SEGFAULT)
        266 - OptiXRapTest.OEventTest (SEGFAULT)
        267 - OptiXRapTest.OInterpolationTest (SEGFAULT)
        268 - OptiXRapTest.ORayleighTest (SEGFAULT)
        271 - OKOPTest.OpIndexerTest (SEGFAULT)
        272 - OKOPTest.OpSeederTest (SEGFAULT)
        273 - OKOPTest.dirtyBufferTest (SEGFAULT)
        274 - OKOPTest.compactionTest (SEGFAULT)
        275 - OKOPTest.OpTest (SEGFAULT)
        277 - OKTest.OKTest (SEGFAULT)
        278 - OKTest.OTracerTest (SEGFAULT)
        280 - OKTest.TrivialTest (SEGFAULT)
        281 - cfg4Test.CMaterialLibTest (SEGFAULT)
        282 - cfg4Test.CMaterialTest (SEGFAULT)
        283 - cfg4Test.CTestDetectorTest (SEGFAULT)
        284 - cfg4Test.CGDMLDetectorTest (SEGFAULT)
        285 - cfg4Test.CGeometryTest (SEGFAULT)
        286 - cfg4Test.CG4Test (SEGFAULT)
        293 - cfg4Test.CCollectorTest (SEGFAULT)
        294 - cfg4Test.CInterpolationTest (SEGFAULT)
        296 - cfg4Test.CGROUPVELTest (SEGFAULT)
        298 - cfg4Test.CPhotonTest (SEGFAULT)
        299 - cfg4Test.CRandomEngineTest (SEGFAULT)
        300 - okg4Test.OKG4Test (SEGFAULT)
    Errors while running CTest
    Wed Jan 10 23:12:40 CST 2018
    opticks-t- : use -V to show output, ctest output written to /usr/local/opticks/build/ctest.log
    delta:~ blyth$ 

