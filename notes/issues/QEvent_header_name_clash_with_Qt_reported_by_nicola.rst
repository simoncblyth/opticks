QEvent_header_name_clash_with_Qt_reported_by_nicola
=====================================================

Nicola reports some vtable issue when using Opticks together with the Qt Geant4 driver, 
presumably due to QEvent.hh header name clash with Qt. Hence have renamed all qudarap/QEvent 
to qudarap/QEvt. 

Before::

    (ok) A[blyth@localhost opticks]$ opticks-fl QEvent
    ./CSGOptiX/CSGOptiX_old.cc
    ./CSGOptiX/Frame.cc
    ./CSGOptiX/cxs_raindrop.sh
    ./CSGOptiX/tests/CSGOptiXSimtraceTest.cc
    ./CSGOptiX/tests/CXRaindropTest.cc
    ./CSGOptiX/cxt_min.sh
    ./CSGOptiX/cxt_precision.sh
    ./CSGOptiX/CSGOptiX.h
    ./CSGOptiX/cxs_min.sh
    ./CSGOptiX/CSGOptiX.cc
    ./cxs_min.sh
    ./cxt_min.sh
    ./g4cx/gxs.sh
    ./g4cx/tests/G4CXTest.sh
    ./g4cx/gxt.sh
    ./g4cx/gxr.sh
    ./qudarap/QEvent.cu
    ./qudarap/tests/CMakeLists.txt
    ./qudarap/tests/QEventTest.cc
    ./qudarap/tests/QEventTest.py
    ./qudarap/tests/QEventTest.sh
    ./qudarap/tests/QEvent_Lifecycle_Test.cc
    ./qudarap/tests/QEvent_Lifecycle_Test.sh
    ./qudarap/tests/QSimTest.cc
    ./qudarap/tests/QSimWithEventTest.cc
    ./qudarap/tests/QSimTest.sh
    ./qudarap/tests/ALL_TEST_runner.sh
    ./qudarap/tests/QEventTest_ALL.sh
    ./qudarap/CMakeLists.txt
    ./qudarap/QSim.cc
    ./qudarap/QU.cc
    ./qudarap/QEvent.hh
    ./qudarap/QSim.hh
    ./qudarap/QEvent.cc
    ./sysrap/SEvent.cc
    ./sysrap/SComp.h
    ./sysrap/SEventConfig.cc
    ./sysrap/SEventConfig.hh
    ./sysrap/SEvt.cc
    ./sysrap/SEvt.hh
    ./sysrap/SGenstep.h
    ./sysrap/salloc.h
    ./sysrap/sevent.h
    ./sysrap/sframe.h
    ./sysrap/sphotonlite.h
    ./u4/tests/U4AppTest.cc
    (ok) A[blyth@localhost opticks]$ 


After::

    (ok) A[blyth@localhost opticks]$ opticks-fl QEvent
    (ok) A[blyth@localhost opticks]



