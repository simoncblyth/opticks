client_server_opticks_testing
================================

TODO
------

* TODO : photon count logging
* TODO : saving gensteps client side for fast replay testing against server without any initialization cost
* TODO : verbosity control on client and server
* TODO : optional optimization loading preexisting Opticks geometry instead of doing the translation ?

  * dangerous as risks using stale Opticks geometry - but non-default-expert-only-option OK

* TODO : loosen requirement for exact same hit flavor between client and server

  * need to signal from client to server via genstep metadata the hit flavor : hit/hitlite/hitlitemerged etc..

* TODO : photon cost aware and VRAM protecting server

  * "Retry-After:?"  header response when server too busy to serve the client
  * how to test this kinda thing ?

* WIP : make OpticksClient release onto cvmfs ?

  * decide on naming and layout of OpticksClient tarball and cvmfs folders and extend okdist-- to do that
  * presumably okdist-- can just detect OPTICKS_CLIENT and act accordingly ?

* TODO : make JUNOSW+OpticksClient release onto cvmfs ?

  * add .gitlab-ci.yml new job to build JUNOSW against OpticksClient
  * see ~/j/oj_client/.gitlab-ci.yml for ideas to avoid duplication


DONE
-----

* DONE : client vs monolithic result comparison, SEvt array saving - EXACT SAME HIT DIGESTS
* FIXED : ~/o/notes/issues/qcerenkov__wavelength_sampled_bndtex_logging_in_server_client_running.rst
* DONE : simple client vs monolithic result comparison - shows issue same with both
* DONE : CK issue not rare, looks like every CK photon afflicted suggesting bnd or matline issue
* DONE : acting on reset in the client
* DONE : added OPTICKS_CONFIG high level control ~/o/notes/issues/generalizing-build-install-dirs-with-OPTICKS_CONFIG.rst
* DONE : build and test JUNOSW against the OPTICKS_CONFIG:Client build "lo_client" (NOT:WITH_CUDA but WITH_CURL, subset of packages + partial packages)



SEvt array saving in server/client vs mono for debug ?
----------------------------------------------------------

* doing this sever side makes sense as that will be almost same as mono runnning ?

Run server with::

    OPTICKS_MODE_SAVE=SMS_RELATIVE OPTICKS_EVENT_MODE=Hit




Test Scripts
---------------

* ~/o/sysrap/tests/SOpticksClientSimulatorTest.sh integrated test


Workflow for client build and test from non-GPU node via reverse tunnel
-------------------------------------------------------------------------

Convenient to keep the four long running processes in separate window:

1. (L) lxlogin soks proxy for github access
2. (A) GPU workstation opticks server
3. (A) "ssh LT" reverse tunnel from A to L, that gives access to A from client on L
4. (L) lxlogin session for running client jobs eg: ljrt100

For development changing client and server make sure to rebuild in appropriate
"lo" or "lo_client" environment - as they have different prefix, despite using
same source tree.


1. Update client Opticks on lxlogin::

   ssh L
   o
   git pull    ## remember to start ssh proxy on lxlogin first "soks-pw-bastion-do"
   lo_client
   oo

2. Update OJ built against client Opticks on lxlogin (not needed when only internal Opticks changes)::

   ssh L
   js
   git pull          # if changes in the junosw branch

   ljclient ljbb0    # update build, no geom init    -- QUICKEST


3. update full Opticks on GPU workstation::

   lo && oo  ## full opticks, update build

4. start server on GPU workstation::

   ~/j/bin/oj_server.sh
   ## AFTER MAKING SERVER CHANGES, HAVE TO BOUNCE THE SERVER
   ## SOME DIR WATCHING IS DONE BUT NOT ENOUGH TO CATCH C++ REBUILDS UPDATING LIBS

5. open reverse tunnel from A (GPU workstation) to L (lxlogin)::

   ssh LT
   ## to avoid exiting this connection, create the reverse tunnel in separate window terminal tab next to the server

6. run test on lxlogin::

   ssh L
   cat ~/.opticks/GEOM/ENVSET.sh # check configured installation (OPTICKS_CONFIG and junosw branch in path to envset.sh)

   ljrt100  # or ljrt1000



Solidify server starting into ~/j/bin/oj_server.sh
---------------------------------------------------

::

    #!/bin/bash
    usage(){ cat << EOU

    ~/j/bin/oj_server.sh

    EOU
    }

    source $HOME/vip/vip.bash
    vip-init-local-base

    source $HOME/j/local.sh

    lo    ## full opticks
    lco   ## miniconda "ok" env with fastapi


    ~/opticks/CSGOptiX/tests/CSGOptiXService_FastAPI_test/CSGOptiXService_FastAPI_test.sh






(A) For comparison of CK issue, mono build on A
------------------------------------------------

::

    lo && oo     ## full opticks, update build
    lo && ENVSET ## config prefix to use to be this non-client build

    ljbb         ## (fresh tab) OJ : ljbb update build with geom init


(A) GPU workstation mono running using the ENVSET configured envset.sh
------------------------------------------------------------------------

Before doing this, stop the server to avoid VRAM issues::

    ljrt100  # or ljrt1000





server-client testing in brief
---------------------------------

*  server : FastAPI python using nanobind to communicate from python to C++ CSGOptiX instance and back

Build and start the server::

    lo  ## (NOT lo_client) - full Opticks env and newer libcurl than system
    ~/opticks/CSGOptiX/tests/CSGOptiXService_FastAPI_test/CSGOptiXService_FastAPI_test.sh


* client 0 : C++ libcurl based using NP_CURL.h NP.hh that repeatedly uploads gensteps loaded from file
* client 1 : curl commandline test client, again uploads gensteps loaded from file

Build and invoke both client tests::

    lo  ## OR "lo_client" - hookup newer curl-config than system one
    ~/np/tests/np_curl_test/np_curl_test.sh


Reverse SSH tunnel to access normally blocked GPU workstation from a login node without GPU
--------------------------------------------------------------------------------------------

On the GPU workstation::

    Host LT
        HostName ip-of-lxlogin
        User blyth
        ## RemoteForward commands defining a Reverse Tunnel back from lxlogin to GPU workstation
        RemoteForward 2222 localhost:22
        # Maps the Workstation SSH port to lxlogin port 2222
        # allowing : "ssh -p 2222 localhost" from lxlogin back to workstation
        RemoteForward 8000 localhost:8000
        # allowing to hit GPU server from lxlogin : curl http://127.0.0.1:8000/docs
        #
        ServerAliveInterval 60
        ExitOnForwardFailure yes
        # ssh -f -N LT
        #    -f: go to background before command execution,
        #    -N: no remote command, just make the tunnel
        # autossh -f -N LT
        #    keep connection alive even when lxlogin drops it
        PermitLocalCommand yes
        LocalCommand echo "LT Tunnel started. To kill it, use: 'pkill -f LT'"



sharing the private tunnel with socat relay ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


socat relay on lxlogin to "bridge" your private tunnel port to a public one::

    # On lxlogin (run by you):
    # This takes traffic from port 8002 and sends it to your private tunnel on 8000
    socat TCP-LISTEN:8002,fork,bind=192.168.x.x TCP:127.0.0.1:8000

::

    A[blyth@localhost opticks]$ sudo dnf search socat --disablerepo=runner_gitlab*
    socat.x86_64 : Bidirectional data relay between two data channels ('netcat++')




L/A : ljrt100 - client/server running : 1711, 1699
-----------------------------------------------------

::

    Begin of Event --> 98
    2026-05-09 15:06:00.869 [junoSD_PMT_v2::EndOfEvent eventID 98 opticksMode 1 hitCollection 0 hitCollectionAlt -1 hcMuon 0 GPU YES
    SOpticksClientSimulator::simulate  eventID 98 reset 0 gs (101, 6, 4, ) hc (1711, 4, 4, ) All/Settings/TreeDigest: YYY dt 0.00922
    2026-05-09 15:06:00.910 ]junoSD_PMT_v2::EndOfEvent eventID 98 opticksMode 1 hitCollection 1711 hitCollectionAlt -1 hcMuon 0 GPU YES
    hitCollectionTT.size: 0	userhitCollectionTT.size: 0
    junotoptask:DetSimAlg.DataModelWriterWithSplit.EndOfEventAction  INFO: writing events with split begin. 2026-05-09 07:06:00.911422000Z
    junotoptask:DetSimAlg.DataModelWriterWithSplit.EndOfEventAction  INFO: writing events with split end. 2026-05-09 07:06:00.917025000Z
    end of event action
    junotoptask:DetSimAlg.execute   INFO: DetSimAlg Simulate An Event (99)
    2026-05-09 15:06:00.918 [junoSD_PMT_v2::Initialize eventID 99
    2026-05-09 15:06:00.919 ]junoSD_PMT_v2::Initialize eventID 99
    Begin of Event --> 99
    2026-05-09 15:06:00.922 [junoSD_PMT_v2::EndOfEvent eventID 99 opticksMode 1 hitCollection 0 hitCollectionAlt -1 hcMuon 0 GPU YES
    SOpticksClientSimulator::simulate  eventID 99 reset 0 gs (138, 6, 4, ) hc (1699, 4, 4, ) All/Settings/TreeDigest: YYY dt 0.00899
    2026-05-09 15:06:00.962 ]junoSD_PMT_v2::EndOfEvent eventID 99 opticksMode 1 hitCollection 1699 hitCollectionAlt -1 hcMuon 0 GPU YES
    hitCollectionTT.size: 0	userhitCollectionTT.size: 0
    junotoptask:DetSimAlg.DataModelWriterWithSplit.EndOfEventAction  INFO: writing events with split begin. 2026-05-09 07:06:00.964250000Z
    junotoptask:DetSimAlg.DataModelWriterWithSplit.EndOfEventAction  INFO: writing events with split end. 2026-05-09 07:06:00.969832000Z
    end of event action
    junotoptask:detsimiotask.finalize  INFO: events processed 100
    junotoptask:DetSimAlg.finalize  INFO: DetSimAlg finalized successfully
    junotoptask:DetSimAlg.InteresingProcessAnaMgr.EndOfRunAction  INFO: All the collected process names:
    junotoptask:DetSimAlg.InteresingProcessAnaMgr.EndOfRunAction  INFO:  -  [ ] eBrem
    junotoptask:DetSimAlg.InteresingProcessAnaMgr.EndOfRunAction  INFO:  -  [ ] eIoni
    junotoptask:DetSimAlg.InteresingProcessAnaMgr.EndOfRunAction  INFO:  -  [ ] phot
    junotoptask:DetSimAlg.InteresingProcessAnaMgr.EndOfRunAction  INFO:  -  [ ] compt
    junotoptask:DetSimAlg.TimerAnaMgr.EndOfRunAction  INFO: summaries:
    junotoptask:DetSimAlg.TimerAnaMgr.EndOfRunAction  INFO: number of measurements: 100
    junotoptask:DetSimAlg.TimerAnaMgr.EndOfRunAction  INFO: mean time: 56.32229 ms
    junotoptask:DetSimAlg.TimerAnaMgr.EndOfRunAction  INFO: total time elapsed: 5632.23682 ms
    ############################## SniperProfiling ##############################
    Name                     Count       Total(ms)      Mean(ms)     RMS(ms)
    GenTools                 100         15.13200       0.15132      0.19700
    DetSimAlg                100         5868.34001     58.68340     36.62761
    Sum of junotoptask       100         5884.06600     58.84066     36.82651
    #############################################################################
    junotoptask:SniperProfiling.finalize  INFO: finalized successfully
    junotoptask:DetSim0Svc.dumpOpticks  INFO: DetSim0Svc::finalizeOpticks m_opticksMode 1 WITH_G4CXOPTICKS
    junotoptask:PMTSimParamSvc.finalize  INFO: PMTSimParamSvc is finalizing!
    junotoptask.finalize            INFO: events processed 100
    [2026-05-09 15:06:01,151] p1244091 {/hpcfs/juno/junogpu/blyth/junosw/InstallArea/Client_blyth-revive-opticksMode-zero-fixing-cleanup-SIGSEGV/python/Tutorial/JUNOApplication.py:225} INFO - ]JUNOAppli


A : ljrt100 - mono running on GPU workstation : 1711, 1699 : same hits num as server/client
----------------------------------------------------------------------------------------------

::

    2026-05-09 12:14:31.936 ]junoSD_PMT_v2::Initialize eventID 98
    Begin of Event --> 98
    2026-05-09 12:14:31.937 [junoSD_PMT_v2::EndOfEvent eventID 98 opticksMode 1 hitCollection 0 hitCollectionAlt -1 hcMuon 0 GPU YES
    SProf::Write DISABLED, enable[export SProf__WRITE=1] disable[unset SProf__WRITE]
    2026-05-09 12:14:31.950 ]junoSD_PMT_v2::EndOfEvent eventID 98 opticksMode 1 hitCollection 1711 hitCollectionAlt -1 hcMuon 0 GPU YES
    hitCollectionTT.size: 0	userhitCollectionTT.size: 0
    junotoptask:DetSimAlg.DataModelWriterWithSplit.EndOfEventAction  INFO: writing events with split begin. 2026-05-09 04:14:31.951062000Z
    junotoptask:DetSimAlg.DataModelWriterWithSplit.EndOfEventAction  INFO: writing events with split end. 2026-05-09 04:14:31.952422000Z
    end of event action
    junotoptask:DetSimAlg.execute   INFO: DetSimAlg Simulate An Event (99)
    2026-05-09 12:14:31.952 [junoSD_PMT_v2::Initialize eventID 99
    2026-05-09 12:14:31.953 ]junoSD_PMT_v2::Initialize eventID 99
    Begin of Event --> 99
    2026-05-09 12:14:31.954 [junoSD_PMT_v2::EndOfEvent eventID 99 opticksMode 1 hitCollection 0 hitCollectionAlt -1 hcMuon 0 GPU YES
    SProf::Write DISABLED, enable[export SProf__WRITE=1] disable[unset SProf__WRITE]
    2026-05-09 12:14:31.967 ]junoSD_PMT_v2::EndOfEvent eventID 99 opticksMode 1 hitCollection 1699 hitCollectionAlt -1 hcMuon 0 GPU YES
    hitCollectionTT.size: 0	userhitCollectionTT.size: 0
    junotoptask:DetSimAlg.DataModelWriterWithSplit.EndOfEventAction  INFO: writing events with split begin. 2026-05-09 04:14:31.968023000Z
    junotoptask:DetSimAlg.DataModelWriterWithSplit.EndOfEventAction  INFO: writing events with split end. 2026-05-09 04:14:31.969373000Z
    end of event action
    junotoptask:detsimiotask.finalize  INFO: events processed 100
    junotoptask:DetSimAlg.finalize  INFO: DetSimAlg finalized successfully
    junotoptask:DetSimAlg.InteresingProcessAnaMgr.EndOfRunAction  INFO: All the collected process names:
    junotoptask:DetSimAlg.InteresingProcessAnaMgr.EndOfRunAction  INFO:  -  [ ] eBrem
    junotoptask:DetSimAlg.InteresingProcessAnaMgr.EndOfRunAction  INFO:  -  [ ] eIoni
    junotoptask:DetSimAlg.InteresingProcessAnaMgr.EndOfRunAction  INFO:  -  [ ] phot
    junotoptask:DetSimAlg.InteresingProcessAnaMgr.EndOfRunAction  INFO:  -  [ ] compt
    junotoptask:DetSimAlg.TimerAnaMgr.EndOfRunAction  INFO: summaries:
    junotoptask:DetSimAlg.TimerAnaMgr.EndOfRunAction  INFO: number of measurements: 100
    junotoptask:DetSimAlg.TimerAnaMgr.EndOfRunAction  INFO: mean time: 17.05153 ms
    junotoptask:DetSimAlg.TimerAnaMgr.EndOfRunAction  INFO: total time elapsed: 1705.15479 ms
    ############################## SniperProfiling ##############################
    Name                     Count       Total(ms)      Mean(ms)     RMS(ms)
    GenTools                 100         5.51900        0.05519      0.11147
    DetSimAlg                100         1767.00500     17.67005     8.95277
    Sum of junotoptask       100         1772.67800     17.72678     9.06424
    #############################################################################
    junotoptask:SniperProfiling.finalize  INFO: finalized successfully
    junotoptask:DetSim0Svc.dumpOpticks  INFO: DetSim0Svc::finalizeOpticks m_opticksMode 1 WITH_G4CXOPTICKS
    junotoptask:PMTSimParamSvc.finalize  INFO: PMTSimParamSvc is finalizing!
    junotoptask.finalize            INFO: events processed 100
    [2026-05-09 12:14:32,015] p3123764 {/home/blyth/junosw/InstallArea/blyth-revive-opticksMode-zero-fixing-cleanup-SIGSEGV/python/Tutorial/JUNOApplication.py:225} INFO - ]JUNOApplication.run
    [2026-05-09 12:14:32,015] p3123764 {/home/blyth/junosw/InstallArea/blyth-revive-opticksMode-zero-fixing-cleanup-SIGSEGV/bin/tut_detsim.py:55} INFO - ]juno_application.run



genstep_hit comparison
-------------------------

::

    A[blyth@localhost oj_release_test]$ ~/np/genstep_hit.sh

       dir         hit     genstep
    -------------------------------
      A000       1,724          90
      A001       1,738         109
      A002       1,711         108
      A003       1,636         114
      A004       1,720         112
      A005       1,657         109
      ...
      A093       1,691         153
      A094       1,793          82
      A095       1,653         129
      A096       1,648         132
      A097       1,729         103
      A098       1,711         101
      A099       1,699         138
    -------------------------------
     TOTAL     171,077      11,243

    Ratio genstep/hit: 0.066
    Shape of counts array: (100, 2)
    A[blyth@localhost oj_release_test]$ pwd
    /tmp/blyth/opticks/oj_release_test




    A[blyth@localhost CSGOptiXService_FastAPI_test]$ ~/np/genstep_hit.sh

       dir         hit     genstep
    -------------------------------
      A000       1,724          90
      A001       1,738         109
      A002       1,711         108
      A003       1,636         114
      A004       1,720         112
      A005       1,657         109
      ...
      A093       1,691         153
      A094       1,793          82
      A095       1,653         129
      A096       1,648         132
      A097       1,729         103
      A098       1,711         101
      A099       1,699         138
    -------------------------------
     TOTAL     171,077      11,243

    Ratio genstep/hit: 0.066
    Shape of counts array: (100, 2)



Getting same hit counts between mono and server/client
--------------------------------------------------------

::

    In [1]: import numpy as np
    In [2]: a = np.load("/data1/blyth/tmp/CSGOptiXService_FastAPI_test/genstep_hit.npy")
    In [3]: b = np.load("/tmp/blyth/opticks/oj_release_test/genstep_hit.npy")
    In [6]: np.all( a == b )
    Out[6]: np.True_


Getting same hit digests between mono and server/client
--------------------------------------------------------

::

    A[blyth@localhost CSGOptiXService_FastAPI_test]$ ~/np/compare_digests.sh /data1/blyth/tmp/CSGOptiXService_FastAPI_test/ALL0_no_opticks_event_name /tmp/blyth/opticks/oj_release_test/ALL0_no_opticks_event_name
    compare_digests.py
    A:        /data1/blyth/tmp/CSGOptiXService_FastAPI_test/ALL0_no_opticks_event_name
    B:        /tmp/blyth/opticks/oj_release_test/ALL0_no_opticks_event_name
    stems:    hit,genstep
    patterns: ['hit.npy', 'genstep.npy']

    file                      digest_A          digest_B    status
    --------------------------------------------------------------
    A000/hit.npy      94108e359c2899bb  94108e359c2899bb        OK
    A000/genstep.npy  c813dc9f47257f6c  c813dc9f47257f6c        OK
    A001/hit.npy      4978f68c7ec56f2a  4978f68c7ec56f2a        OK
    A001/genstep.npy  0c204073cce1bd83  0c204073cce1bd83        OK
    A002/hit.npy      0c81a034589a30a4  0c81a034589a30a4        OK
    A002/genstep.npy  2def732f39c49c5d  2def732f39c49c5d        OK
    A003/hit.npy      f2df20f6fdf8c781  f2df20f6fdf8c781        OK
    A003/genstep.npy  32341d7dcdaee71a  32341d7dcdaee71a        OK
    A004/hit.npy      7a7c7ce829d3cb4b  7a7c7ce829d3cb4b        OK
    A004/genstep.npy  242ef58a4cc2b066  242ef58a4cc2b066        OK
    A005/hit.npy      b946ebae5cb07159  b946ebae5cb07159        OK
    ...
    A097/genstep.npy  07da6431eac8f496  07da6431eac8f496        OK
    A098/hit.npy      319d86dab51e29d6  319d86dab51e29d6        OK
    A098/genstep.npy  44f1073d50e62d03  44f1073d50e62d03        OK
    A099/hit.npy      d1eac4f078611c39  d1eac4f078611c39        OK
    A099/genstep.npy  25b89d837d41eec4  25b89d837d41eec4        OK
    --------------------------------------------------------------

    Compared 100 dirs × 2 arrays = 200 files
    Only in A: 0 dirs
    Only in B: 0 dirs

    All digests match.



