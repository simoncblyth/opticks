classification_of_some_tests_eg_when_looking_for_template_for_new_tests
=========================================================================




om build CUDA tests using "extern" to glue together functionality
-----------------------------------------------------------------------


Good example of mechanics, (without getting enormous like QSim)::

    qudarap/QPMT* 
    qudarap/tests/QPMTTest.sh

::

    [blyth@localhost opticks]$ find . -name '*.cc' -exec grep -l extern\  {} \;
    ./examples/UseFindOpticks/tests/DemoLibTest.cc
    ./examples/UseNVRTC/UseNVRTC.cc
    ./examples/UseNVRTC/UseNVRTC2.cc
    ./qudarap/QBnd.cc
    ./qudarap/QCerenkov.cc
    ./qudarap/QCurandState.cc
    ./qudarap/QEvent.cc
    ./qudarap/QMultiFilm.cc
    ./qudarap/QOptical.cc
    ./qudarap/QPMT.cc
    ./qudarap/QPoly.cc
    ./qudarap/QProp.cc
    ./qudarap/QScint.cc
    ./qudarap/QSim.cc
    ./qudarap/QSim_dbg.cc
    ./qudarap/QTexLookup.cc
    ./qudarap/QTexRotate.cc
    ./qudarap/QRng.cc




gcc + nvcc tests standalone
------------------------------

::
    
   ./qudarap/tests/QPMT_Test.sh:        nvcc -c $cui \


Optix ptx/optixir tests
---------------------------

::

    ./sysrap/tests/SGLFW_SOPTIX_Scene_test.sh:    nvcc $cu \
    ./sysrap/tests/SOPTIX_Scene_test.sh:   nvcc $cu \
            optix tests, with ptx/xir loading 


Very Simple single .cu tests
-------------------------------

::

    ./sysrap/tests/SIMGStandaloneTest.sh:nvcc $name.cu \
    ./sysrap/tests/SUTest.sh:nvcc -c ../SU.cu -I.. -o $dir/SU.o
    ./sysrap/tests/erfcinvf_Test.sh:    nvcc $name.cu -std=c++11 $opt -I$HOME/np -I..  -I$CUDA_PREFIX/include -o $bin
    ./sysrap/tests/logTest.sh:    nvcc $name.cu -std=c++11 $opt -I.. -I/usr/local/cuda/include -o /tmp/$name 
    ./thrustrap/tests/strided_rangeTest.sh:nvcc $name.cu -I.. -I../../sysrap -o $bin
          these there are too small, .cu only 


