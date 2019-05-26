cluster-slurm-ctest-fails
==============================


::

    #! /bin/bash -l

    #SBATCH --partition=gpu
    #SBATCH --qos=debug
    #SBATCH --account=junogpu
    #SBATCH --job-name=scb
    #SBATCH --ntasks=1
    #SBATCH --output=/hpcfs/juno/junogpu/blyth/out/%j.out
    #SBATCH --mem-per-cpu=20480
    #SBATCH --gres=gpu:v100:8

    ...

    oxrap-
    oxrap-bcd
    cd tests
    which ctest
    ctest --output-on-failure



Hmm all failing to load optix libs::

    bench.py --name geocache-machinery
    Namespace(digest=None, exclude=None, include=None, metric='launchAVG', name='geocache-machinery', other='prelaunch000', resultsdir='$OPTICKS_RESULTS_PREFIX/results', since=None)
    ()
    bench.py --name geocache-machinery
    /afs/ihep.ac.cn/users/b/blyth/g/local/bin/ctest
    Test project /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/optixrap/tests
          Start  1: OptiXRapTest.OContextCreateTest
     1/19 Test  #1: OptiXRapTest.OContextCreateTest ..............***Failed    0.01 sec 
    /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/optixrap/tests/OContextCreateTest: error while loading shared libraries: liboptix.so.6.0.0: cannot open shared object file: No such file or directory

          Start  2: OptiXRapTest.OScintillatorLibTest
     2/19 Test  #2: OptiXRapTest.OScintillatorLibTest ............***Failed    0.01 sec 
    /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/optixrap/tests/OScintillatorLibTest: error while loading shared libraries: liboptix.so.6.0.0: cannot open shared object file: No such file or directory

          Start  3: OptiXRapTest.LTOOContextUploadDownloadTest
     3/19 Test  #3: OptiXRapTest.LTOOContextUploadDownloadTest ...***Failed    0.01 sec 
    /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/optixrap/tests/LTOOContextUploadDownloadTest: error while loading shared libraries: liboptix.so.6.0.0: cannot open shared object file: No such file or directory



    
::

    # Note that ctest running uses the test executables from the build tree
    # not the installed executables which find libs via their RPATH.
    # Normally ctest manages to arrange that build tree test executables find their libs,
    # but somehow this fails to work with slurm batch running.
    # The workaround is to temporarily set the library path to find optix libs. 

    LD_LIBRARY_PATH=$LOCAL_BASE/opticks/externals/OptiX/lib64 ctest --output-on-failure



