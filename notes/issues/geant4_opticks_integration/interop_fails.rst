
Interop Failures
============================


Interop mode with seed buffer : get hard crash even with trivial
-------------------------------------------------------------------

::

    simon:opticks blyth$ OKTest --trivial --save 

    2016-09-13 18:33:42.255 INFO  [47199] [OContext::getFormat@555] OContext::getFormat override format for seed 
    2016-09-13 18:33:42.255 INFO  [47199] [OContext::configureBuffer@498]       seed          100000,1,1SEED size 100000
    2016-09-13 18:33:42.257 INFO  [47199] [OEvent::upload@200] OEvent::upload (INTEROP) gensteps handed to OptiX by referencing OpenGL buffer id  
    2016-09-13 18:33:42.257 INFO  [47199] [OEvent::upload@204] OEvent::upload DONE
    2016-09-13 18:33:42.257 INFO  [47199] [OpSeeder::seedPhotonsFromGensteps@52] OpSeeder::seedPhotonsFromGensteps
    2016-09-13 18:33:42.257 INFO  [47199] [OpSeeder::seedPhotonsFromGenstepsViaOpenGL@67] OpSeeder::seedPhotonsFromGenstepsViaOpenGL
    iexpand  counts_size 1 output_size 100000
    2016-09-13 18:33:42.315 INFO  [47199] [OContext::close@224] OContext::close numEntryPoint 2
    2016-09-13 18:33:42.569 INFO  [47199] [OContext::launch@250] OContext::launch entry 0 width 0 height 0


::

    OKTest --nopropagate   # doesnt crash when skip the launch 






Trying to do raytrace when loaded, fails for lack of record buffer
------------------------------------------------------------------------

::

    OKTest --load   ## then press O,shift-O


    2016-09-09 17:02:10.832 INFO  [136214] [Interactor::key_pressed@428] Interactor::key_pressed O nextRenderStyle 
    Renderer::update_uniforms ClipPlane
               1.000            0.000            0.000        18079.453 
    Interactor::key_pressed 340 
    2016-09-09 17:02:12.765 INFO  [136214] [Interactor::key_pressed@428] Interactor::key_pressed O nextRenderStyle 
    2016-09-09 17:02:12.781 INFO  [136214] [OTracer::trace_@123] OTracer::trace  entry_index 1 trace_count 0 resolution_scale 1 size(2880,1704) ZProj.zw (-1.04082,-288.615) front 0.7071,0.7071,0.0000
    2016-09-09 17:02:12.781 INFO  [136214] [OContext::close@195] OContext::close numEntryPoint 2
    2016-09-09 17:02:13.070 INFO  [136214] [OContext::launch@221] OContext::launch entry 1 width 2880 height 1704
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Invalid value (Details: Function "RTresult _rtContextValidate(RTcontext)" caught exception: Non-initialized variable record_buffer:  Buffer(1d, 8 byte element), file:/Users/umber/workspace/rel4.0-mac64-build-Release/sw/wsapps/raytracing/rtsdk/rel4.0/src/Context/ValidationManager.cpp, line: 118)
    Abort trap: 6
    simon:ggeoview blyth$ 



Record buffer is special, as it is a non-interop buffer, it is written by OptiX and 
read from Thrust to create recsel and phosel, but it is never available to OpenGL. 
So OpticksViz::uploadEvent will not upload it.


::

    simon:ggeoview blyth$ OKTest --load --dbguploads 


    2016-09-09 17:22:44.460 INFO  [144121] [Opticks::makeEvent@591] Opticks::makeEvent G4  tagoffset 0 id 0
    2016-09-09 17:22:44.460 INFO  [144121] [OpticksEvent::setBufferControl@708]    nopstep : (spec) : INTEROP_MODE 
    2016-09-09 17:22:44.460 INFO  [144121] [OpticksEvent::setBufferControl@708]     photon : (spec) : OPTIX_INPUT_OUTPUT PTR_FROM_OPENGL INTEROP_MODE 
    2016-09-09 17:22:44.460 INFO  [144121] [OpticksEvent::setBufferControl@708]   sequence : (spec) : OPTIX_NON_INTEROP OPTIX_OUTPUT_ONLY INTEROP_MODE 
    2016-09-09 17:22:44.460 INFO  [144121] [OpticksEvent::setBufferControl@708]     phosel : (spec) : INTEROP_MODE 
    2016-09-09 17:22:44.460 INFO  [144121] [OpticksEvent::setBufferControl@708]     recsel : (spec) : INTEROP_MODE 
    2016-09-09 17:22:44.460 INFO  [144121] [OpticksEvent::setBufferControl@708]     record : (spec) : OPTIX_OUTPUT_ONLY INTEROP_MODE 
    2016-09-09 17:22:44.460 INFO  [144121] [*Opticks::makeEvent@591] Opticks::makeEvent OK  tagoffset 0 id 1
    2016-09-09 17:22:44.460 INFO  [144121] [OpticksEvent::setBufferControl@708]    nopstep : (spec) : INTEROP_MODE 
    2016-09-09 17:22:44.460 INFO  [144121] [OpticksEvent::setBufferControl@708]     photon : (spec) : OPTIX_INPUT_OUTPUT PTR_FROM_OPENGL INTEROP_MODE 
    2016-09-09 17:22:44.461 INFO  [144121] [OpticksEvent::setBufferControl@708]   sequence : (spec) : OPTIX_NON_INTEROP OPTIX_OUTPUT_ONLY INTEROP_MODE 
    2016-09-09 17:22:44.461 INFO  [144121] [OpticksEvent::setBufferControl@708]     phosel : (spec) : INTEROP_MODE 
    2016-09-09 17:22:44.461 INFO  [144121] [OpticksEvent::setBufferControl@708]     recsel : (spec) : INTEROP_MODE 
    2016-09-09 17:22:44.461 INFO  [144121] [OpticksEvent::setBufferControl@708]     record : (spec) : OPTIX_OUTPUT_ONLY INTEROP_MODE 
    2016-09-09 17:22:44.461 INFO  [144121] [Parameters::set@117] Parameters::set changing TimeStamp from 20160909_172244 to 20160909_172244
    2016-09-09 17:22:44.461 INFO  [144121] [Report::load@48] Report::load from /tmp/blyth/opticks/evt/dayabay/torch/1/report.txt
    2016-09-09 17:22:44.462 INFO  [144121] [OpticksEvent::importParameters@1413] OpticksEvent::importParameters  mode_ COMPUTE_MODE --> COMPUTE_MODE
    2016-09-09 17:22:44.757 INFO  [144121] [OpticksEvent::loadBuffers@1580] OpticksEvent::load shape(0) before reshaping  num_genstep 1 num_nopstep 0 [  num_photons 100000 num_history 100000 num_phosel 100000 ]  [  num_records 100000 num_recsel 100000 ] 
    2016-09-09 17:22:44.757 INFO  [144121] [OpticksEvent::setBufferControl@708]    genstep : (spec) : OPTIX_INPUT_ONLY UPLOAD_WITH_CUDA BUFFER_COPY_ON_DIRTY COMPUTE_MODE 
    2016-09-09 17:22:44.757 INFO  [144121] [OpticksEvent::setBufferControl@708]    nopstep : (spec) : COMPUTE_MODE 
    2016-09-09 17:22:44.757 INFO  [144121] [OpticksEvent::setBufferControl@708]     photon : (spec) : OPTIX_INPUT_OUTPUT PTR_FROM_OPENGL COMPUTE_MODE 
    2016-09-09 17:22:44.767 INFO  [144121] [OpticksEvent::setBufferControl@708]   sequence : (spec) : OPTIX_NON_INTEROP OPTIX_OUTPUT_ONLY COMPUTE_MODE 
    2016-09-09 17:22:44.773 INFO  [144121] [OpticksEvent::setBufferControl@708]     record : (spec) : OPTIX_OUTPUT_ONLY COMPUTE_MODE 
    2016-09-09 17:22:44.877 INFO  [144121] [OpticksEvent::setBufferControl@708]     phosel : (spec) : COMPUTE_MODE 
    2016-09-09 17:22:44.880 INFO  [144121] [OpticksEvent::setBufferControl@708]     recsel : (spec) : COMPUTE_MODE 
    2016-09-09 17:22:44.906 INFO  [144121] [OpticksEvent::loadBuffers@1611] OpticksEvent::load  genstep 1,6,4 nopstep 0,4,4 photon 100000,4,4 record 100000,10,2,4 phosel 100000,1,4 recsel 100000,10,1,4 sequence 100000,1,2
    2016-09-09 17:22:44.906 INFO  [144121] [Composition::setCenterExtent@991] Composition::setCenterExtent ce -18079.4531,-799699.4375,-6605.0000,1000.0000
    2016-09-09 17:22:44.907 INFO  [144121] [OpticksHub::target@437] OpticksHub::target evt  typ torch tag 1 itag 1 det dayabay cat  dir /tmp/blyth/opticks/evt/dayabay/torch/1 eng OK gsce -18079.4531,-799699.4375,-6605.0000,1000.0000
    2016-09-09 17:22:44.907 INFO  [144121] [OpticksViz::uploadEvent@277] OpticksViz::uploadEvent (1)
    2016-09-09 17:22:44.908 INFO  [144121] [Rdr::upload@303]       axis_attr vpos cn        3 sh                3,3,4 id    21 dt   0x7ff0dbd0a3c0 hd     Y nb        144 GL_STATIC_DRAW
    2016-09-09 17:22:44.910 INFO  [144121] [Rdr::upload@303]    genstep_attr vpos cn        1 sh                1,6,4 id    22 dt   0x7ff0dd367820 hd     Y nb         96 GL_STATIC_DRAW
    2016-09-09 17:22:44.913 INFO  [144121] [Rdr::upload@303]    nopstep_attr vpos cn        0 sh                0,4,4 id    23 dt              0x0 hd     N nb          0 GL_STATIC_DRAW
    2016-09-09 17:22:44.915 INFO  [144121] [Rdr::upload@303]     photon_attr vpos cn   100000 sh           100000,4,4 id    24 dt      0x133ab4000 hd     Y nb    6400000 GL_DYNAMIC_DRAW
    2016-09-09 17:22:44.932 INFO  [144121] [Rdr::upload@303]     record_attr rpos cn  1000000 sh        100000,10,2,4 id    25 dt      0x135012000 hd     Y nb   16000000 GL_STATIC_DRAW
    2016-09-09 17:22:44.971 INFO  [144121] [Rdr::upload@303]   sequence_attr phis cn   100000 sh           100000,1,2 id    26 dt      0x1360dc000 hd     Y nb    1600000 GL_STATIC_DRAW
    2016-09-09 17:22:44.973 INFO  [144121] [Rdr::upload@303]     phosel_attr psel cn   100000 sh           100000,1,4 id    27 dt      0x1362c5000 hd     Y nb     400000 GL_STATIC_DRAW
    2016-09-09 17:22:44.974 INFO  [144121] [Rdr::upload@303]     recsel_attr rsel cn  1000000 sh        100000,10,1,4 id    28 dt      0x136327000 hd     Y nb    4000000 GL_STATIC_DRAW
    2016-09-09 17:22:44.978 INFO  [144121] [Scene::dump_uploads_table@782] OpticksViz::uploadEvent(--dbguploads)
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@132] photon Rdr tag: pos
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@144]     photon_attr 0/ 4 vnpy       vpos    100000 npy 100000,4,4 npy.hasData 1
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@144]     photon_attr 1/ 4 vnpy       vdir    100000 npy 100000,4,4 npy.hasData 1
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@144]     photon_attr 2/ 4 vnpy       vpol    100000 npy 100000,4,4 npy.hasData 1
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@144]     photon_attr 3/ 4 vnpy       iflg    100000 npy 100000,4,4 npy.hasData 1
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@144]   sequence_attr 0/ 2 vnpy       phis    100000 npy 100000,1,2 npy.hasData 1
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@144]   sequence_attr 1/ 2 vnpy       pmat    100000 npy 100000,1,2 npy.hasData 1
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@144]     phosel_attr 0/ 1 vnpy       psel    100000 npy 100000,1,4 npy.hasData 1
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@132] record Rdr tag: rec
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@144]     record_attr 0/ 4 vnpy       rpos   1000000 npy 100000,10,2,4 npy.hasData 1
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@144]     record_attr 1/ 4 vnpy       rpol   1000000 npy 100000,10,2,4 npy.hasData 1
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@144]     record_attr 2/ 4 vnpy       rflg   1000000 npy 100000,10,2,4 npy.hasData 1
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@144]     record_attr 3/ 4 vnpy       rflq   1000000 npy 100000,10,2,4 npy.hasData 1
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@144]     recsel_attr 0/ 1 vnpy       rsel   1000000 npy 100000,10,1,4 npy.hasData 1
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@132] altrecord Rdr tag: altrec
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@144]     record_attr 0/ 4 vnpy       rpos   1000000 npy 100000,10,2,4 npy.hasData 1
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@144]     record_attr 1/ 4 vnpy       rpol   1000000 npy 100000,10,2,4 npy.hasData 1
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@144]     record_attr 2/ 4 vnpy       rflg   1000000 npy 100000,10,2,4 npy.hasData 1
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@144]     record_attr 3/ 4 vnpy       rflq   1000000 npy 100000,10,2,4 npy.hasData 1
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@144]     recsel_attr 0/ 1 vnpy       rsel   1000000 npy 100000,10,1,4 npy.hasData 1
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@132] devrecord Rdr tag: devrec
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@144]     record_attr 0/ 4 vnpy       rpos   1000000 npy 100000,10,2,4 npy.hasData 1
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@144]     record_attr 1/ 4 vnpy       rpol   1000000 npy 100000,10,2,4 npy.hasData 1
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@144]     record_attr 2/ 4 vnpy       rflg   1000000 npy 100000,10,2,4 npy.hasData 1
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@144]     record_attr 3/ 4 vnpy       rflq   1000000 npy 100000,10,2,4 npy.hasData 1
    2016-09-09 17:22:44.978 INFO  [144121] [Rdr::dump_uploads_table@144]     recsel_attr 0/ 1 vnpy       rsel   1000000 npy 100000,10,1,4 npy.hasData 1
    2016-09-09 17:22:44.978 INFO  [144121] [OpticksViz::uploadEvent@284] OpticksViz::uploadEvent (1) DONE 
    2016-09-09 17:22:44.978 INFO  [144121] [OpticksViz::indexPresentationPrep@311] OpticksViz::indexPresentationPrep
    2016-09-09 17:22:44.981 INFO  [144121] [GPropertyLib::close@316] GPropertyLib::close type GBndLib buf 123,4,2,39,4
    2016-09-09 17:22:44.981 INFO  [144121] [Bookmarks::create@249] Bookmarks::create : persisting state to slot 0
    2016-09-09 17:22:44.981 INFO  [144121] [Bookmarks::collect@273] Bookmarks::collect 0
    2016-09-09 17:22:44.984 INFO  [144121] [OpticksViz::renderLoop@431] enter runloop 
    2016-09-09 17:22:45.000 INFO  [144121] [OpticksViz::renderLoop@436] after frame.show() 
    Frame::handle_event window resized to (0 0)


Problem may be that the OptiX buffers are only created when OEvent::upload is called which isnt happening on load.

  




Multi event testing CUDA memory error
----------------------------------------


::
    OKTest 

    ...
    2016-09-08 21:03:56.200 INFO  [3537] [OContext::configureBuffer@432]   sequence          100000,1,2 USER size (ijk)     200000 elementsize 8
    2016-09-08 21:03:56.203 INFO  [3537] [SLog::operator@15] OEvent::OEvent DONE
    2016-09-08 21:03:56.203 INFO  [3537] [OpSeeder::seedPhotonsFromGensteps@61] OpSeeder::seedPhotonsFromGensteps
    2016-09-08 21:03:56.203 INFO  [3537] [OpSeeder::seedPhotonsFromGenstepsViaOpenGL@76] OpSeeder::seedPhotonsFromGenstepsViaOpenGL
    2016-09-08 21:03:56.240 INFO  [3537] [OpSeeder::seedPhotonsFromGenstepsImp@148] OpSeeder::seedPhotonsFromGenstepsImp gensteps 1,6,4 num_genstep_values 24
    iexpand  counts_size 1 output_size 100000
    2016-09-08 21:03:56.260 INFO  [3537] [OpZeroer::zeroRecords@54] OpZeroer::zeroRecords
    OpZeroer::zeroRecordsViaOpenGL(CBufSpec)s_rec : dev_ptr 0x711040000 size 8000000 num_bytes 16000000 hexdump 0 
    2016-09-08 21:03:56.274 INFO  [3537] [OContext::launch@221] OContext::launch entry 0 width 100000 height 1
    libc++abi.dylib: terminating with uncaught exception of type optix::Exception: Memory allocation failed (Details: Function "RTresult _rtContextLaunch2D(RTcontext, unsigned int, RTsize, RTsize)" caught exception: Encountered a CUDA error: driver().cuGraphicsMapResources( 1, &m_resource, hStream.get() ) returned (2): Out of memory, file:/Users/umber/workspace/rel4.0-mac64-build-Release/sw/wsapps/raytracing/rtsdk/rel4.0/src/CUDA/GraphicsResource.cpp, line: 73)
    Abort trap: 6
    simon:opticks blyth$ 
    simon:opticks blyth$ 

