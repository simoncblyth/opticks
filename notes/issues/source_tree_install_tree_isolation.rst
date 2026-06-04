source_tree_install_tree_isolation
====================================


CSGOptiX CMake installs the scripts::

    -- Up-to-date: /data1/blyth/local/opticks_Debug/bin/cxr_min.sh
    -- Up-to-date: /data1/blyth/local/opticks_Debug/bin/cxr_min_debug.sh
    -- Up-to-date: /data1/blyth/local/opticks_Debug/bin/cxt_min.sh
    -- Up-to-date: /data1/blyth/local/opticks_Debug/bin/cxt_min.py
    -- Up-to-date: /data1/blyth/local/opticks_Debug/bin/cxs_min.sh
    -- Up-to-date: /data1/blyth/local/opticks_Debug/bin/cxs_min.py
    -- Up-to-date: /data1/blyth/local/opticks_Debug/bin/cxs_min_AB.py
    -- Up-to-date: /data1/blyth/local/opticks_Debug/bin/cxs_min_lite.py


PATH is setup to find the installed ones::

    [lo] A[blyth@localhost CSGOptiX]$ which cxs_min.sh
    /data1/blyth/local/opticks_Debug/bin/cxs_min.sh

    [lo] A[blyth@localhost CSGOptiX]$ which cxs_min.py
    /data1/blyth/local/opticks_Debug/bin/cxs_min.py

    [lo] A[blyth@localhost CSGOptiX]$ which CSGOptiXSMTest
    /data1/blyth/local/opticks_Debug/lib/CSGOptiXSMTest



To zeroth order, the difference between source and install running is simple::

    ~/o/cxs_min.sh   ## source tree running
    cxs_min.sh       ## install tree running

But having HOME in PYTHONPATH mixes the trees::

    [lo] A[blyth@localhost CSGOptiX]$ echo $PYTHONPATH | tr ":" "\n"
    /home/blyth                            ## OOPS - WHERE DOES THIS COME FROM ?
    /data1/blyth/local/opticks_Debug/py



Where is HOME included into PYTHONPATH ?
------------------------------------------

::

    A[blyth@localhost opticks]$ echo $PYTHONPATH

    A[blyth@localhost opticks]$ lo
    ..
    [lo] A[blyth@localhost opticks]$ echo $PYTHONPATH
    /home/blyth:/data1/blyth/local/opticks_Debug/py



::

    [local_ok_externals] A[blyth@localhost opticks]$ t local_ok_build
    local_ok_build () 
    { 
        : ~/j/local.sh;
        : ~/j/opticks_config.sh which sets OPTICKS_CONFIG was formerly sourced after local_ok_externals;
        : BUT now with the Client config which changes the required externals;
        : the ordering is switched;
        source ~/j/opticks_config.sh;
        local_ok_externals;
        om-;
        [ $? -ne 0 ] && echo $BASH_SOURCE $FUNCNAME - ENV SETUP FAIL NEW NODES OR BUILD CONFIGS REQUIRE opticks-full ONCE && return 1;
        : REQUIREMENT TO opticks-full FOR EVERY CONFIG IS BECAUSE CONFIG STRING IS PART OF THE PREFIX AND MANAGED EXTERNALS ARE WITHIN THE PREFIX DIR;
        export VIP_MODE=ok_build;
        export VIP_DESC="env to build Opticks against JUNOSW externals manually setup from cvmfs : Xercesc + CLHEP + Geant4 + custom4 + Python + python-numpy";
        function ok_build () 
        { 
            cd ~/opticks;
            om
        };
        type ok_build;
        : noticed that need to om- to get setup
    }



BINGO::

    A[blyth@localhost opticks]$ source ~/j/opticks_config.sh
    A[blyth@localhost opticks]$ echo $PYTHONPATH
    /home/blyth



Server environment setup
--------------------------

HMM the below misses externals setup::

    source /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-vLatest/bashrc

Generate convenience envset.sh for standalone Opticks OK running just like have for OJ running,
so can::

    source /cvmfs/opticks.ihep.ac.cn/ok/releases/el9_amd64_gcc11/Opticks-vLatest/envset.sh
    cxs_min.sh


