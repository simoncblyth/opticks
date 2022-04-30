remove-OpticksCore-dependency-from-CSGOptiX
=============================================

SBT.cc using SolidSelection vector from Opticks : relocate where ?
----------------------------------------------------------------------

* inside CSGFoundry would seem the natural place as it is the primary user 


::

    epsilon:CSGOptiX blyth$ opticks-f getSolidSelection
    ./CSGOptiX/SBT.cc:1. Opticks::getSolidSelection
    ./CSGOptiX/SBT.cc:    solid_selection(ok->getSolidSelection()),   // vector<unsigned>
    ./CSGOptiX/tests/CSGOptiXRenderTest.cc:    solid_selection(ok->getSolidSelection()), //  NB its not set yet, that happens below 
    ./CSGOptiX/Six.cc:    solid_selection(ok->getSolidSelection()),
    ./sysrap/SOpticks.hh:    std::vector<unsigned>&        getSolidSelection() ;
    ./sysrap/SOpticks.hh:    const std::vector<unsigned>&  getSolidSelection() const ;
    ./sysrap/SOpticks.cc:std::vector<unsigned>&  SOpticks::getSolidSelection() 
    ./sysrap/SOpticks.cc:const std::vector<unsigned>&  SOpticks::getSolidSelection() const 
    ./optickscore/Opticks.hh:       std::vector<unsigned>&  getSolidSelection() ; 
    ./optickscore/Opticks.hh:       const std::vector<unsigned>& getSolidSelection() const ;
    ./optickscore/Opticks.cc:std::vector<unsigned>& Opticks::getSolidSelection()
    ./optickscore/Opticks.cc:const std::vector<unsigned>& Opticks::getSolidSelection() const 



* vector is populated based on solid_label argument, that solid_label can instead come in via envvar  

::

     88 CSGOptiXRenderTest::CSGOptiXRenderTest(int argc, char** argv)
     89     :
     90     ok(InitOpticks(argc, argv)),
     91     solid_label(ok->getSolidLabel()),         // --solid_label   used for selecting solids from the geometry 
     92     solid_selection(ok->getSolidSelection()), //  NB its not set yet, that happens below 
     93     fd(CSGFoundry::Load()),
     94     cx(nullptr),


::

    epsilon:CSG blyth$ opticks-f findSolidIdx
    ./CSGOptiX/tests/CSGOptiXRenderTest.cc:        fd->findSolidIdx(solid_selection, solid_label); 
    ./CSG/CSGFoundry.h:    int findSolidIdx(const char* label) const  ; // -1 if not found
    ./CSG/CSGFoundry.h:    void findSolidIdx(std::vector<unsigned>& solid_idx, const char* label) const ; 
    ./CSG/tests/CMakeLists.txt:    CSGFoundry_findSolidIdx_Test.cc
    ./CSG/tests/CSGFoundry_findSolidIdx_Test.cc:void test_findSolidIdx(const CSGFoundry* fd, int argc, char** argv)
    ./CSG/tests/CSGFoundry_findSolidIdx_Test.cc:        fd->findSolidIdx(solid_selection, sla );   
    ./CSG/tests/CSGFoundry_findSolidIdx_Test.cc:    test_findSolidIdx(fd, argc, argv); 
    ./CSG/CSGFoundry.cc:int CSGFoundry::findSolidIdx(const char* label) const 
    ./CSG/CSGFoundry.cc:CSGFoundry::findSolidIdx
    ./CSG/CSGFoundry.cc:void CSGFoundry::findSolidIdx(std::vector<unsigned>& solid_idx, const char* label) const 
    ./CSG/CSGFoundry.cc:    findSolidIdx(solidIdx, label); 
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ 



