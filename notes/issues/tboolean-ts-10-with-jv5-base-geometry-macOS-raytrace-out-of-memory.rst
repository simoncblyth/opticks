tboolean-ts-10-with-jv5-base-geometry-macOS-raytrace-out-of-memory
====================================================================


ts 10 then O to switch to raytrace::


    2019-06-25 16:39:46.138 INFO  [18475246] [Scene::setRecordStyle@1127] line
    2019-06-25 16:39:46.671 INFO  [18475246] [Scene::setRecordStyle@1127] vector
    2019-06-25 16:39:50.538 INFO  [18475246] [RenderStyle::setRenderStyle@99] RenderStyle R_COMPOSITE
    2019-06-25 16:39:50.543 INFO  [18475246] [OTracer::trace_@140]  entry_index 1 trace_count 0 resolution_scale 1 pixeltime_scale 1000 size(2880,1704) ZProj.zw (-1.04082,-14693.9) front -0.8198,-0.5589,0.1244
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Memory allocation failed (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: cudaDriver().CuGraphicsUnmapResources( count, resources, hStream.get() ) returned (2): Out of memory)
    /Users/blyth/opticks/bin/o.sh: line 234: 26538 Abort trap: 6           /usr/local/opticks/lib/OKG4Test --okg4 --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero --envkey --rendermode +global,+axis --geocenter --stack 2180 --eye 1,0,0 --test --testconfig autoseqmap=TO:0,SR:1,SA:0_name=tboolean-proxy-10_outerfirst=1_analytic=1_csgpath=/tmp/blyth/opticks/tboolean-proxy-10_mode=PyCsgInBox_autoobject=Vacuum/perfectSpecularSurface//GlassSchottF2_autoemitconfig=photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55,diffuse:1,ctmindiffuse:0.5,ctmaxdiffuse:1.0_autocontainer=Rock//perfectAbsorbSurface/Vacuum --torch --torchconfig type=disc_photons=100000_mode=fixpol_polarization=1,1,0_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000_source=0,



This repeats and recurs after a reboot. 


