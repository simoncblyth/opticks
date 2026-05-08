client_server_opticks_testing
================================

TODO
------

* WIP : acting on reset in the client, verbosity control on client and server

* TODO : loosen requirement for exact same hit flavor between client and server - need to signal from client to server
  via genstep metadata the hit flavor : hit/hitlite/hitlitemerged etc..

* WIP : make OpticksClient release onto cvmfs ?

  * decide on naming and layout of OpticksClient tarball and cvmfs folders and extend okdist-- to do that
  * presumably okdist-- can just detect OPTICKS_CLIENT and act accordingly ?

* TODO : make JUNOSW+OpticksClient release onto cvmfs ?

  * add .gitlab-ci.yml new job to build JUNOSW against OpticksClient
  * see ~/j/oj_client/.gitlab-ci.yml for ideas to avoid duplication


DONE
-----

* DONE : added OPTICKS_CONFIG high level control ~/o/notes/issues/generalizing-build-install-dirs-with-OPTICKS_CONFIG.rst
* DONE : build and test JUNOSW against the OPTICKS_CONFIG:Client build "lo_client" (NOT:WITH_CUDA but WITH_CURL, subset of packages + partial packages)


Test Scripts
---------------

* ~/o/sysrap/tests/SOpticksClientSimulatorTest.sh integrated test



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




