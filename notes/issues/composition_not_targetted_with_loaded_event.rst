FIXED : Composition Not Targetted with Loaded Event
======================================================

The Fix
---------

* https://bitbucket.org/simoncblyth/opticks/commits/eecbf120e8be


Former Problem
----------------

In GGeoViewTest without "--load" the targgeting is based
on the loaded genstep tests/GGeoViewTest.cc::

    098     if(!nooptix && !load)
     99     {
    100         app.loadGenstep();             // hostside load genstep into NumpyEvt
    101 
    102         app.targetViz();               // point Camera at gensteps 


But in "--load" mode the genstep is never loaded, so falls back to geo-targetting.
This is problematic for comparison of compute and interop mode simulations.


Approach : persist gensteps together with evt and use for targetting
-----------------------------------------------------------------------

Reason is that the gensteps were not persisted with the event, as they are 
somehow different in that they come from elsewhere. Which is the case with cerenkov 
and scintillation, but with torch the gensteps are just created there and then.
Now that input gensteps have a well defined home, can just start writing 
gensteps together with an evt for posterity, especially for the derived TORCH 
gensteps.

::

    simon:boostrap blyth$ BOpticksResourceTest
    BOpticksResource::Summary
    prefix   : /usr/local/opticks
    envprefix: OPTICKS_
    opticksdata_dir      /usr/local/opticks/opticksdata
    resource_dir         /usr/local/opticks/opticksdata/resource
    gensteps_dir         /usr/local/opticks/opticksdata/gensteps
    installcache_dir     /usr/local/opticks/installcache
    rng_installcache_dir /usr/local/opticks/installcache/RNG
    okc_installcache_dir /usr/local/opticks/installcache/OKC
    ptx_installcache_dir /usr/local/opticks/installcache/PTX
    getPTXPath(generate.cu.ptx) = /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate.cu.ptx
    PTXPath(generate.cu.ptx) = /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate.cu.ptx
    simon:boostrap blyth$ 


::

    simon:issues blyth$ find /usr/local/opticks/opticksdata/gensteps -type f -exec du -h {} \;
    736K    /usr/local/opticks/opticksdata/gensteps/dayabay/cerenkov/1.npy
    1.3M    /usr/local/opticks/opticksdata/gensteps/dayabay/scintillation/1.npy
    364K    /usr/local/opticks/opticksdata/gensteps/juno/cerenkov/1.npy
    168K    /usr/local/opticks/opticksdata/gensteps/juno/scintillation/1.npy





