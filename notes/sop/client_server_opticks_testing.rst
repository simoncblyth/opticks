client_server_opticks_testing
================================

TODO
------

* TODO : photon count logging
* TODO : saving gensteps client slide for fast replay testing against server without any initialization cost
* TODO : client vs monolithic result comparison, SEvt array saving
* TODO : investigate "//qcerenkov::wavelength_sampled_bndtex" logging on server - rare or significant ? photon count logging

  * ~/o/notes/issues/qcerenkov__wavelength_sampled_bndtex_logging_in_server_client_running.rst 

* TODO : verbosity control on client and server
* TODO : optional optimization loading preexisting Opticks geometry instead of doing the translation ?

  * dangerous as risks using stale Opticks geometry - but non-default-expert-only-option OK

* TODO : loosen requirement for exact same hit flavor between client and server - need to signal from client to server
  via genstep metadata the hit flavor : hit/hitlite/hitlitemerged etc..

* TODO : photon cost aware and VRAM protecting server, "Retry-After:?"  header response when server too busy

* WIP : make OpticksClient release onto cvmfs ?

  * decide on naming and layout of OpticksClient tarball and cvmfs folders and extend okdist-- to do that
  * presumably okdist-- can just detect OPTICKS_CLIENT and act accordingly ?

* TODO : make JUNOSW+OpticksClient release onto cvmfs ?

  * add .gitlab-ci.yml new job to build JUNOSW against OpticksClient
  * see ~/j/oj_client/.gitlab-ci.yml for ideas to avoid duplication


DONE
-----

* DONE : acting on reset in the client
* DONE : added OPTICKS_CONFIG high level control ~/o/notes/issues/generalizing-build-install-dirs-with-OPTICKS_CONFIG.rst
* DONE : build and test JUNOSW against the OPTICKS_CONFIG:Client build "lo_client" (NOT:WITH_CUDA but WITH_CURL, subset of packages + partial packages)


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


3. start server on GPU workstation::

   lo   ## full opticks
   lco  ## need "fastapi" from miniconda "ok" python virtual env
   ~/opticks/CSGOptiX/tests/CSGOptiXService_FastAPI_test/CSGOptiXService_FastAPI_test.sh
   ## AFTER MAKING SERVER CHANGES, REMEMBER TO BOUNCE THE SERVER

4. open reverse tunnel from A (GPU workstation) to L (lxlogin)::

   ssh LT
   ## to avoid exiting this connection, create the reverse tunnel in separate window terminal tab next to the server

5. run test on lxlogin::

   ssh L
   cat ~/.opticks/GEOM/ENVSET.sh # check configured installation (OPTICKS_CONFIG and junosw branch in path to envset.sh)

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




