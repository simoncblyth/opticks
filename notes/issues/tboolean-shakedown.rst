tboolean-shakedown
=====================



g4 : TO NA MI : FIXED 
----------------------------

::

    ts truncate
    ts cone
    ts zsphere1


    ts prism --noalign
    ts icosahedron --noalign
    ts zsphere2 --noalign
    ts union-zsphere --noalign
    ts difference-zsphere --noalign


* :doc:`tboolean-g4-TO-NA-MI`


zsphere1 : different geometry ! translation issue ? FIXED
-----------------------------------------------------------------

Open Opticks viz in two sessions and start animations in each, shows
clearly different geometry::

   tv zsphere1
   tv4 zsphere1

  
* OK : intended big cheese shape
* G4 : back to back cones 

* :doc:`tboolean-zsphere1-zsphere2-discrep`

HOW TO PROCEED : intersect full G4Orb with a suitable box to get the 
z-slicing that Opticks zsphere is intended to be   

* fixed with CMaker::ConvertZSphere


CG4 CRandomEngine::flat processName
------------------------------------------

* calls to G4UniformRand which get routed via engine expected to be done by a process
* probably everything that does not use emit=-1 or emit=1 will show this

* using emitters and input photons is really convenient for debugging, because of 
  the masked photon support allowing to rerun single (or a few) photons : so  
  probably simplest to just configure emitters for the tests 

* nevertheless will have to solve this for testing of genstep running 

::

    ts prism 
    ts icosahedron
    ts trapezoid
    ts union-zsphere
    ts difference-zsphere
    ts zsphere2 

::

    Program received signal SIGSEGV, Segmentation fault.
    0x00007fffe2978c90 in std::string::c_str() const () from /lib64/libstdc++.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glfw-3.2.1-2.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libX11-1.6.5-2.el7.x86_64 libX11-devel-1.6.5-2.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXcursor-1.1.15-1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libXfixes-5.0.3-1.el7.x86_64 libXinerama-1.1.3-2.1.el7.x86_64 libXrandr-1.5.1-2.el7.x86_64 libXrender-0.9.10-1.el7.x86_64 libXxf86vm-1.1.4-1.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libdrm-2.4.91-3.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 xerces-c-3.1.1-9.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffe2978c90 in std::string::c_str() const () from /lib64/libstdc++.so.6
    #1  0x00007fffefdf3fcb in CRandomEngine::CurrentProcessName () at /home/blyth/opticks/cfg4/CRandomEngine.cc:174
    #2  0x00007fffefdf42e7 in CRandomEngine::flat (this=0x72caea0) at /home/blyth/opticks/cfg4/CRandomEngine.cc:228
    #3  0x00007fffec3ddec0 in G4SPSRandomGenerator::GenRandX (this=0x7505300) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/event/src/G4SPSRandomGenerator.cc:255
    #4  0x00007fffec3d4f03 in G4SPSPosDistribution::GeneratePointsInPlane (this=0x7505ce0, pos=...) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/event/src/G4SPSPosDistribution.cc:403
    #5  0x00007fffec3db0dc in G4SPSPosDistribution::GenerateOne (this=0x7505ce0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/event/src/G4SPSPosDistribution.cc:1178
    #6  0x00007fffefde0620 in CTorchSource::GeneratePrimaryVertex (this=0x7504930, event=0xe440530) at /home/blyth/opticks/cfg4/CTorchSource.cc:285
    #7  0x00007fffefdbf2e0 in CPrimaryGeneratorAction::GeneratePrimaries (this=0x75067a0, event=0xe440530) at /home/blyth/opticks/cfg4/CPrimaryGeneratorAction.cc:15
    #8  0x00007fffec6abba7 in G4RunManager::GenerateEvent (this=0x72cb670, i_event=0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:460
    #9  0x00007fffec6ab63c in G4RunManager::ProcessOneEvent (this=0x72cb670, i_event=0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:398
    #10 0x00007fffec6ab4d7 in G4RunManager::DoEventLoop (this=0x72cb670, n_event=10, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:367
    #11 0x00007fffec6aad2d in G4RunManager::BeamOn (this=0x72cb670, n_event=10, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:273
    #12 0x00007fffefde9f9c in CG4::propagate (this=0x72cb0d0) at /home/blyth/opticks/cfg4/CG4.cc:335
    #13 0x00007ffff7bd570f in OKG4Mgr::propagate_ (this=0x7fffffffcdf0) at /home/blyth/opticks/okg4/OKG4Mgr.cc:177
    #14 0x00007ffff7bd55cf in OKG4Mgr::propagate (this=0x7fffffffcdf0) at /home/blyth/opticks/okg4/OKG4Mgr.cc:117
    #15 0x00000000004039a9 in main (argc=33, argv=0x7fffffffd128) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:9


Not instanciating CRandomEngine avoids the "flat" issue::

    ts prism --noalign    
    ts icosahedron --noalign
    ts trapezoid --noalign


CRandomEngine OUT OF RANGE
----------------------------

::

    ts cubeplanes


CRandomEngine sequence
-----------------------------

::

    ts uniontree
    ts zsphere2 
  

::

    OKG4Test: /home/blyth/opticks/cfg4/CRandomEngine.cc:296: double CRandomEngine::_flat(): Assertion `m_cursor >= 0 && m_cursor < int(m_sequence.size())' failed.

Avoid issue with::

    ts uniontree --noalign 
    ts zsphere2 --noalign

        

TO AB : photons start in Rock ?
--------------------------------------

::

    ts trapezoid --noalign


material inconsistency
----------------------------

* breaks the Russian Doll assumption 

::

    ts parade
    ts complement


::

    2019-06-24 20:36:28.793 FATAL [409880] [npart::check_bb_zero@172] check_bb_zero endcap flags expected 3 (ignored anyhow) 0
    2019-06-24 20:36:29.164 FATAL [409880] [NCSGList::checkMaterialConsistency@333]  BOUNDARY IMAT/OMAT INCONSISTENT  bparent.imat != bself.omat  i 2 numTree 11
                bparent                             Vacuum///GlassSchottF2       bparent.imat        GlassSchottF2



ts boxrot : 1 degree rotated about X : 3 photons scatter occurs at same place but different directions
------------------------------------------------------------------------------------------------------------------

* curiously varying the angle and the problem comes and goes

::

    tboolean-boxrot--(){ cat << EOP 
    import logging
    log = logging.getLogger(__name__)
    from opticks.ana.main import opticks_main
    from opticks.analytic.polyconfig import PolyConfig
    from opticks.analytic.csg import CSG  
    from opticks.analytic.glm import rotate  


    autoemitconfig="photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55,diffuse:1,ctmindiffuse:0.5,ctmaxdiffuse:1.0"
    args = opticks_main(csgname="${FUNCNAME/--}", autoemitconfig=autoemitconfig)

    emitconfig = "photons:100000,wavelength:380,time:0.0,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55" 

    CSG.kwa = dict(poly="IM",resolution=20, verbosity=0, ctrl=0, containerscale=3.0, emitconfig=emitconfig  )

    container = CSG("box", emit=-1, boundary='Rock//perfectAbsorbSurface/Vacuum', container=1 )  # no param, container=1 switches on auto-sizing

    box = CSG("box3", param=[300,300,200,0], emit=0,  boundary="Vacuum///GlassSchottF2"  )

    #degrees = 45
    degrees = 1
    box.transform = rotate([1,0,0,degrees])

    CSG.Serialize([container, box], args )
    EOP
    }




::

    ab.ox_dv
    maxdvmax:910.3344  level:FATAL  RC:1       skip:
                     :                                :                   :                       :                   : 0.1000 0.2500 0.5000 :                                    
      idx        msg :                            sel :    lcu1     lcu2  :       nitem     nelem :  nwar  nerr  nfat :   fwar   ferr   ffat :        mx        mn       avg      
     0000            :                    TO BT BT SA :    8808     8808  :        8808    105696 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :                 INFO :  
     0001            :                       TO BR SA :     580      580  :         580      6960 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :                 INFO :  
     0002            :                 TO BT BR BT SA :     562      562  :         562      6744 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :                 INFO :  
     0003            :              TO BT BR BR BT SA :      29       29  :          29       348 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :                 INFO :  
     0004            :           TO BT BR BR BR BT SA :       6        6  :           6        72 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :                 INFO :  
     0005            :                       TO SC SA :       5        5  :           5        60 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0001    0.0000    0.0000   :                 INFO :  
     0006            :                 TO BT BT SC SA :       3        3  :           3        36 :    18    18    13 : 0.5000 0.5000 0.3611 :  910.3344    0.0000   75.0998   :                FATAL :   > dvmax[2] 0.5000  
     0007            :                       TO BT AB :       2        2  :           2        24 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :                 INFO :  
     0008            :                 TO BT SC BT SA :       2        2  :           2        24 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0000    0.0000    0.0000   :                 INFO :  
     0009            :           TO BT BT SC BT BT SA :       1        1  :           1        12 :     5     4     4 : 0.4167 0.3333 0.3333 :  262.7971    0.0000   23.8921   :                FATAL :   > dvmax[2] 0.5000  
     0010            :              TO BR SC BT BT SA :       1        1  :           1        12 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0001    0.0000    0.0000   :                 INFO :  
     0011            :              TO SC BT BR BT SA :       1        1  :           1        12 :     0     0     0 : 0.0000 0.0000 0.0000 :    0.0001    0.0000    0.0000   :                 INFO :  
    .





Two GUI comparison selecting "TO BT BT SC SA", looks like scatters are happening at same position but into different directions::

   tv boxrot
   tv4 boxrot


Confirmed by the numbers, just the last SA point are different::

    In [2]: a.rpost()
    Out[2]: 
    A()sliced
    A([[[  45.426 ,   10.0374, -455.0674,    0.    ],
        [  45.426 ,   10.0374,  -99.8453,    1.185 ],
        [  45.426 ,    8.6592,  100.1655,    2.397 ],
        [  45.426 ,    8.6592,  166.3903,    2.6178],
        [ 455.1648, -175.133 ,  342.1637,    4.2263]],

       [[   9.7451,   -4.4827, -455.0674,    0.    ],
        [   9.7451,   -4.4827, -100.0959,    1.1842],
        [   9.7451,   -5.861 ,   99.9149,    2.3962],
        [   9.7451,   -5.861 ,  412.2865,    3.4381],
        [ -17.6386,  -14.1025,  455.1648,    3.6099]],

       [[  -5.7635,   23.7084, -455.0674,    0.    ],
        [  -5.7635,   23.7084,  -99.5947,    1.1858],
        [  -5.7635,   22.3162,  100.4022,    2.3976],
        [  -5.7635,   22.3162,  222.9535,    2.8066],
        [-455.1648, -151.0627,  445.6564,    4.5766]]])

    In [3]: b.rpost()
    Out[3]: 
    A()sliced
    A([[[  45.426 ,   10.0374, -455.0674,    0.    ],
        [  45.426 ,   10.0374,  -99.8453,    1.185 ],
        [  45.426 ,    8.6592,  100.1655,    2.397 ],
        [  45.426 ,    8.6592,  166.3903,    2.6178],
        [-455.1648,  233.1998,  381.1301,    4.583 ]],

       [[   9.7451,   -4.4827, -455.0674,    0.    ],
        [   9.7451,   -4.4827, -100.0959,    1.1842],
        [   9.7451,   -5.861 ,   99.9149,    2.3962],
        [   9.7451,   -5.861 ,  412.2865,    3.4381],
        [  37.1148,    2.3667,  455.1648,    3.6099]],

       [[  -5.7635,   23.7084, -455.0674,    0.    ],
        [  -5.7635,   23.7084,  -99.5947,    1.1858],
        [  -5.7635,   22.3162,  100.4022,    2.3976],
        [  -5.7635,   22.3162,  222.9535,    2.8066],
        [ 455.1648,  200.15  ,  451.3642,    4.622 ]]])

    In [4]: a.rpost() - b.rpost()
    Out[4]: 
    A()sliced
    A([[[   0.    ,    0.    ,    0.    ,    0.    ],
        [   0.    ,    0.    ,    0.    ,    0.    ],
        [   0.    ,    0.    ,    0.    ,    0.    ],
        [   0.    ,    0.    ,    0.    ,    0.    ],
        [ 910.3296, -408.3328,  -38.9664,   -0.3567]],

       [[   0.    ,    0.    ,    0.    ,    0.    ],
        [   0.    ,    0.    ,    0.    ,    0.    ],
        [   0.    ,    0.    ,    0.    ,    0.    ],
        [   0.    ,    0.    ,    0.    ,    0.    ],
        [ -54.7534,  -16.4692,    0.    ,    0.    ]],

       [[   0.    ,    0.    ,    0.    ,    0.    ],
        [   0.    ,    0.    ,    0.    ,    0.    ],
        [   0.    ,    0.    ,    0.    ,    0.    ],
        [   0.    ,    0.    ,    0.    ,    0.    ],
        [-910.3296, -351.2127,   -5.7078,   -0.0454]]])

    In [5]: 







