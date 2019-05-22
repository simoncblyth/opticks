quartic_solve_optix_600_misaligned_address_exception
========================================================

Context
----------

* :doc:`OptiX_600_CUDA_10.1_test_fails`


geocache-j1808
------------------

OKX4Test parses the gdml, creates geocache, pops up OpenGL gui, 
switching to ray trace works but as soon as navigate into region where torus is needed
get the Misaligned address issue, presumably quartic double problem.

Torus strikes, see notes/issues/torus_replacement_on_the_fly.rst for the fix::

    [blyth@localhost issues]$ geocache-j1808
    geocache-j1808 is a function
    geocache-j1808 () 
    { 
        type \$FUNCNAME;
        opticksdata-;
        OKX4Test --gdmlpath \$(opticksdata-j) --g4codegen --csgskiplv 22
    }
    2019-04-15 10:45:36.211 INFO  [150689] [main@74]  parsing /home/blyth/local/opticks/opticksdata/export/juno1808/g4_00.gdml
    G4GDML: Reading '/home/blyth/local/opticks/opticksdata/export/juno1808/g4_00.gdml'...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    ...

    019-04-15 10:47:27.086 FATAL [150689] [ContentStyle::setContentStyle@98] ContentStyle norm inst 1 bbox 0 wire 0 asis 0 m_num_content_style 0 NUM_CONTENT_STYLE 5
    2019-04-15 10:47:32.590 INFO  [150689] [RenderStyle::setRenderStyle@95] RenderStyle R_COMPOSITE
    2019-04-15 10:47:32.820 INFO  [150689] [OTracer::trace_@128] OTracer::trace  entry_index 0 trace_count 0 resolution_scale 1 size(1920,1080) ZProj.zw (-1.04082,-17316.9) front 0.5824,0.8097,-0.0719
    terminate called after throwing an instance of 'optix::Exception'
      what():  Unknown error (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: cudaDriver().CuEventSynchronize( m_event ) returned (716): Misaligned address)
    ^CKilled
    [blyth@localhost issues]$ 



Thoughts
----------

Suspect this will take a long time to fix : it is the double heavy GPU quartic solving.
Another guise of the long standing torus issue striking again.

So perhaps, as only have TITAN RTX for a few more days, and want to 
make some full geometry RTX benchmarks better to :doc:`torus_replacement_on_the_fly`





