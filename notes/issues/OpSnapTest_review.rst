OpSnapTest review : compute only snapshots of geometry saved to PPM files
==========================================================================


Issues
----------

1. compute snaps are vertically flipped compared to those from Frame::snap (oglrap Pix)
2. seems that the snap configi moving more than intended 

   * the problem was a mismatch between the default snapconfig and the initial eye position

3. crazy undefined polarization setting : it doesnt matter as its just tracing geometry :
   but where is it coming from ?

   * fixed by initializing to some defaults in GenstepNPY/TorchstepNPY

4. BTree is failing to load ChromaMaterialMap.json : is that still needed ? The path 
   is from opticksdata : surely that cannot be right in direct mode : nothing 
   should come from opticksdata ?

     

::

   CUDA_VISIBLE_DEVICES=1 OPTICKS_RTX=0 OpSnapTest --envkey --target 352851 --eye -1,-1,-1 --snapconfig "numsteps=5,eyestartz=-1,eyestopz=-0.5" 





OpSnapTest : compute only snaps : based on OpMgr which is used only by G4Opticks and OpSnapTest 
---------------------------------------------------------------------------------------------------------

::

    int main(int argc, char** argv)
    {
        OPTICKS_LOG(argc, argv);
        Opticks ok(argc, argv, "--tracer"); 
        OpMgr op(&ok);
        op.snap();
        return 0 ; 
    }

    155 void OpMgr::snap()
    156 {
    157     LOG(info) << "OpMgr::snap" ;
    158     m_propagator->snap();
    159 }

    106 void OpPropagator::snap()
    107 {
    108     LOG(info) << "OpPropagator::snap" ;
    109     m_tracer->snap();
    110 }


     92 /**
     93 OpTracer::snap
     94 ----------------
     95  
     96 Takes one or more GPU raytrace snapshots of geometry
     97 at various positions configured via m_snap_config.  
     98  
     99 **/
    100  
    101 void OpTracer::snap()
    102 {
    103     LOG(info) << "OpTracer::snap START" ;
    104     m_snap_config->dump();
    105  
    106     int num_steps = m_snap_config->steps ;
    107     float eyestartz = m_snap_config->eyestartz ;
    108     float eyestopz = m_snap_config->eyestopz ;
    109  
    110     for(int i=0 ; i < num_steps ; i++)
    111     {
    112         std::string path = m_snap_config->getSnapPath(i) ;
    113  
    114         float frac = num_steps > 1 ? float(i)/float(num_steps-1) : 0.f ;
    115         float eyez = eyestartz + (eyestopz-eyestartz)*frac ;
    116  
    117         std::cout << " i " << std::setw(5) << i
    118                   << " eyez " << std::setw(10) << eyez
    119                   << " path " << path
    120                   << std::endl ;
    121  
    122         m_composition->setEyeZ( eyez );
    123  
    124         render();
    125  
    126         m_ocontext->snap(path.c_str());
    127     }
    128  
    129     LOG(info) << "OpTracer::snap DONE " ;
    130 }

    079 void OpTracer::render()
     80 {
     81     if(m_count == 0 )
     82     {
     83         m_hub->setupCompositionTargetting();
     84         m_otracer->setResolutionScale(1) ;
     85     }
     86  
     87     m_otracer->trace_();
     88     m_count++ ;
     89 }


Launch times are collected into m_trace_times STimes instance held in OTracer 
with sums compile/prelaunch/launch times and counts calls (so effectively average timings over all snaps).
::

    284 void OContext::launch(unsigned int lmode, unsigned int entry, unsigned int width, unsigned int height, STimes* times )
    285 {
    286     if(!m_closed) close();
    287 
    288     LOG(LEVEL)
    289               << " entry " << entry
    290               << " width " << width
    291               << " height " << height
    292               ;
    293 
    294     if(times) times->count     += 1 ;
    295 
    296     if(lmode & VALIDATE)
    297     {
    298         double dt = validate_();
    299         LOG(LEVEL) << "VALIDATE time: " << dt ;
    300         if(times) times->validate  += dt  ;
    301     }
    302 
    303     if(lmode & COMPILE)
    304     {
    305         double dt = compile_();
    306         LOG(LEVEL) << "COMPILE time: " << dt ;
    307         if(times) times->compile  += dt ;
    308     }
    309 
    310     if(lmode & PRELAUNCH)
    311     {
    312         double dt = launch_(entry, width, height );
    313         LOG(LEVEL) << "PRELAUNCH time: " << dt ;
    314         if(times) times->prelaunch  += dt ;
    315     }
    316 
    317     if(lmode & LAUNCH)
    318     {
    319         double dt = m_llogpath ? launch_redirected_(entry, width, height ) : launch_(entry, width, height );
    320         LOG(LEVEL) << "LAUNCH time: " << dt  ;
    321         if(times) times->launch  += dt  ;
    322     }
    323 }
    324 



::

    OpSnapTest --envkey --xanalytic --target 10000
    ...






Undefined Polarization
------------------------


::

    2019-04-21 18:48:35.035 ERROR [160701] [OpticksGen::makeTorchstep@373]  as torchstep isDefault replacing placeholder frame  frameIdx : 0 detectorDefaultFrame : 0
    2019-04-21 18:48:35.035 INFO  [160701] [OpticksGen::targetGenstep@303] OpticksGen::targetGenstep setting frame 0 1.0000,0.0000,0.0000,0.0000 0.0000,1.0000,0.0000,0.0000 0.0000,0.0000,1.0000,0.0000 0.0000,0.0000,0.0000,1.0000
    2019-04-21 18:48:35.035 FATAL [160701] [GenstepNPY::setPolarization@262] GenstepNPY::setPolarization pol 0.0000,281494486304241768988672.0000,0.0000,0.0000 npol 0.0000,0.0000,0.0000,0.0000 m_polw 0.0000,0.0000,0.0000,430.0000
    OpSnapTest: /home/blyth/opticks/npy/GenstepNPY.cpp:268: void GenstepNPY::setPolarization(const vec4&): Assertion `0' failed.

    Program received signal SIGABRT, Aborted.
    0x00007fffec06a207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    0  0x00007fffec06a207 in raise () from /lib64/libc.so.6
    1  0x00007fffec06b8f8 in abort () from /lib64/libc.so.6
    2  0x00007fffec063026 in __assert_fail_base () from /lib64/libc.so.6
    3  0x00007fffec0630d2 in __assert_fail () from /lib64/libc.so.6
    4  0x00007ffff3adf14e in GenstepNPY::setPolarization (this=0x57e4020, pol=...) at /home/blyth/opticks/npy/GenstepNPY.cpp:268
    5  0x00007ffff3add514 in TorchStepNPY::update (this=0x57e4020) at /home/blyth/opticks/npy/TorchStepNPY.cpp:469
    6  0x00007ffff3adea4d in GenstepNPY::addStep (this=0x57e4020, verbose=false) at /home/blyth/opticks/npy/GenstepNPY.cpp:135
    7  0x00007ffff5fd95ae in OpticksGen::makeTorchstep (this=0x57e3ef0) at /home/blyth/opticks/opticksgeo/OpticksGen.cc:388
    8  0x00007ffff5fd8770 in OpticksGen::makeLegacyGensteps (this=0x57e3ef0, code=4096) at /home/blyth/opticks/opticksgeo/OpticksGen.cc:207
    9  0x00007ffff5fd8552 in OpticksGen::initFromLegacyGensteps (this=0x57e3ef0) at /home/blyth/opticks/opticksgeo/OpticksGen.cc:171
    10 0x00007ffff5fd7f36 in OpticksGen::init (this=0x57e3ef0) at /home/blyth/opticks/opticksgeo/OpticksGen.cc:109
    11 0x00007ffff5fd7e05 in OpticksGen::OpticksGen (this=0x57e3ef0, hub=0x63e540) at /home/blyth/opticks/opticksgeo/OpticksGen.cc:50
    12 0x00007ffff5fd3408 in OpticksHub::init (this=0x63e540) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:234
    13 0x00007ffff5fd3198 in OpticksHub::OpticksHub (this=0x63e540, ok=0x7fffffffd740) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:195
    14 0x00007ffff7b56aad in OpMgr::OpMgr (this=0x7fffffffd6d0, ok=0x7fffffffd740) at /home/blyth/opticks/okop/OpMgr.cc:46
    15 0x0000000000402cb4 in main (argc=8, argv=0x7fffffffd9d8) at /home/blyth/opticks/okop/tests/OpSnapTest.cc:26
    (gdb) f 12
    12 0x00007ffff5fd3408 in OpticksHub::init (this=0x63e540) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:234
    234     m_gen = new OpticksGen(this) ;
    (gdb) f 11
    11 0x00007ffff5fd7e05 in OpticksGen::OpticksGen (this=0x57e3ef0, hub=0x63e540) at /home/blyth/opticks/opticksgeo/OpticksGen.cc:50
    50      init() ;
    (gdb) f 10
    10 0x00007ffff5fd7f36 in OpticksGen::init (this=0x57e3ef0) at /home/blyth/opticks/opticksgeo/OpticksGen.cc:109
    109         initFromLegacyGensteps();
    (gdb) f 9
    9  0x00007ffff5fd8552 in OpticksGen::initFromLegacyGensteps (this=0x57e3ef0) at /home/blyth/opticks/opticksgeo/OpticksGen.cc:171
    171     NPY<float>* gs = makeLegacyGensteps(code) ; 
    (gdb) f 8
    8  0x00007ffff5fd8770 in OpticksGen::makeLegacyGensteps (this=0x57e3ef0, code=4096) at /home/blyth/opticks/opticksgeo/OpticksGen.cc:207
    207         m_torchstep = makeTorchstep() ;
    (gdb) f 7
    7  0x00007ffff5fd95ae in OpticksGen::makeTorchstep (this=0x57e3ef0) at /home/blyth/opticks/opticksgeo/OpticksGen.cc:388
    388     torchstep->addStep(torchdbg);  // copyies above configured step settings into the NPY and increments the step index, ready for configuring the next step 
    (gdb) f 6
    6  0x00007ffff3adea4d in GenstepNPY::addStep (this=0x57e4020, verbose=false) at /home/blyth/opticks/npy/GenstepNPY.cpp:135
    135     update(); 
    (gdb) f 5
    5  0x00007ffff3add514 in TorchStepNPY::update (this=0x57e4020) at /home/blyth/opticks/npy/TorchStepNPY.cpp:469
    469     setPolarization(pol); 
    (gdb) p pol
    $1 = {{x = 2.03623588e-19, r = 2.03623588e-19, s = 2.03623588e-19}, {y = 2.81494486e+23, g = 2.81494486e+23, t = 2.81494486e+23}, {z = 1.35867427e-19, b = 1.35867427e-19, p = 1.35867427e-19}, {
        w = 1.9600338e-19, a = 1.9600338e-19, q = 1.9600338e-19}}
    (gdb) 


::

    452 void TorchStepNPY::update()
    453 {
    454    // direction from: target - source
    455    // position from : source
    456 
    457     const glm::mat4& frame_transform = getFrameTransform() ;
    458 
    459     m_src = frame_transform * m_source_local  ;
    460     m_tgt = frame_transform * m_target_local  ;
    461     glm::vec4 pol = frame_transform * m_polarization_local  ;   // yields unnormalized, but GenstepNPY setter normalizes
    462 
    463     m_dir = glm::vec3(m_tgt) - glm::vec3(m_src) ;
    464 
    465     glm::vec3 dir = glm::normalize( m_dir );
    466 
    467     setPosition(m_src);
    468     setDirection(dir);
    469     setPolarization(pol);
    470 
    471 }




ChromaMaterialMap.json legacy file still being looked for by OpticksHub with OpSnapTest 
-----------------------------------------------------------------------------------------------

::

    2019-04-21 19:13:09.776 INFO  [210071] [BOpticksKey::SetKey@45] BOpticksKey::SetKey from OPTICKS_KEY envvar OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
    2019-04-21 19:13:09.786 WARN  [210071] [BTree::loadTree@50] BTree.loadTree: can't find file /home/blyth/local/opticks/opticksdata/export/OKX4Test/ChromaMaterialMap.json
    OpSnapTest: /home/blyth/opticks/boostrap/BTree.cc:53: static int BTree::loadTree(boost::property_tree::ptree&, const char*): Assertion `0' failed.

    Program received signal SIGABRT, Aborted.
    0x00007fffec06a207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    0  0x00007fffec06a207 in raise () from /lib64/libc.so.6
    1  0x00007fffec06b8f8 in abort () from /lib64/libc.so.6
    2  0x00007fffec063026 in __assert_fail_base () from /lib64/libc.so.6
    3  0x00007fffec0630d2 in __assert_fail () from /lib64/libc.so.6
    4  0x00007ffff3596e74 in BTree::loadTree (t=..., path=0x648d48 "/home/blyth/local/opticks/opticksdata/export/OKX4Test/ChromaMaterialMap.json") at /home/blyth/opticks/boostrap/BTree.cc:53
    5  0x00007ffff35590f2 in BMap<std::string, unsigned int>::load (this=0x7fffffffd090, path=0x63a610 "/home/blyth/local/opticks/opticksdata/export/OKX4Test/ChromaMaterialMap.json", depth=0)
        at /home/blyth/opticks/boostrap/BMap.cc:150
    6  0x00007ffff35588c6 in BMap<std::string, unsigned int>::load (mp=0x7fffffffd0c0, path=0x63a610 "/home/blyth/local/opticks/opticksdata/export/OKX4Test/ChromaMaterialMap.json", depth=0)
        at /home/blyth/opticks/boostrap/BMap.cc:58
    7  0x00007ffff5fd4696 in OpticksHub::configureLookupA (this=0x63e540) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:453
    8  0x00007ffff5fd3367 in OpticksHub::init (this=0x63e540) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:218
    9  0x00007ffff5fd3198 in OpticksHub::OpticksHub (this=0x63e540, ok=0x7fffffffd740) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:195
    10 0x00007ffff7b56aad in OpMgr::OpMgr (this=0x7fffffffd6d0, ok=0x7fffffffd740) at /home/blyth/opticks/okop/OpMgr.cc:46
    11 0x0000000000402cb4 in main (argc=8, argv=0x7fffffffd9d8) at /home/blyth/opticks/okop/tests/OpSnapTest.cc:26
    (gdb) 


Snap running needs to be classified as embedded ?::

    (gdb) f 8 
    #8  0x00007ffff5fd3367 in OpticksHub::init (this=0x63e540) at /home/blyth/opticks/opticksgeo/OpticksHub.cc:218
    218     if(!m_ok->isEmbedded()) configureLookupA();
    (gdb) p m_ok->isEmbedded()
    $1 = false
    (gdb) 


Need to shakedown gensteps in direct mode, to resolve this.

