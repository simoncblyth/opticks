lxslc-opticks-build-for-okdist-cvmfs-testing
===============================================


* GPU cluster NVIDIA driver still at 418.40.04 so cannot go from OptiX 6.0.0 to 6.5.0 yet  


New external::

   opticksaux-
   opticksaux--
    
Moved glew install::

   glew-
   glew-lib64-rm
   glew--
   o
   om-cleaninstall oglrap:

Three new subs, as scripts etc.. need to be installed for okdist-- tarballing::

    blyth@localhost opticks]$ om-subs
    okconf
    sysrap
    boostrap
    npy
    yoctoglrap
    optickscore
    ggeo
    assimprap
    openmeshrap
    opticksgeo
    cudarap
    thrustrap
    optixrap
    okop
    oglrap
    opticksgl
    ok
    extg4
    cfg4
    okg4
    g4ok
    integration
    ana                  ##
    analytic             ##
    bin                  ##


Also require cleaninstall (not actually needed as above one did it already)::

   o
   om-cleaninstall ana:


Create tarball::


