how-to-find-a-shy-photon-propagation-visualization
========================================================

Overview
----------

Look at the event numerically to debug why not visible.



Rerun
--------

Increase logging of relevant classes and save event data::

    OpticksAim=INFO OpticksGen=INFO OKTest --compute --save --dbgaim

Forcing the target gets vtk_oxplt.py to give expected AD cylinder of final positions::

    OpticksAim=INFO OpticksGen=INFO OKTest --compute --save --dbgaim --gensteptarget 3154


0. look at gs.npy
-------------------

Zeros show are aiming at geocenter, which in DYB geom is nowhere::

    In [1]: gs = np.load("gs.npy")                                                                                                                                         

    In [3]: np.set_printoptions(suppress=True)                                                                                                                             

    In [4]: gs                                                                                                                                                             
    Out[4]: 
    array([[[  0. ,   0. ,   0. ,   0. ],
            [  0. ,   0. ,   0. ,   0.1],
            [  0. ,   0. ,   1. ,   1. ],
            [  0. ,   0. ,   1. , 430. ],
            [  0. ,   1. ,   0. ,   1. ],
            [  0. ,   0. ,   0. ,   0. ]]], dtype=float32)



1. save the event and take a look at ox.npy
-----------------------------------------------



Add "--compute"  and "--save" to the commandline to download it from GPU and save to file::

    geocache-gui()
    {
       local dbg 
       [ -n "$DBG" ] && dbg="gdb --args" || dbg=""
       CUDA_VISIBLE_DEVICES=1 OKTest --envkey \
                    --xanalytic \
                    --compute \
                    --save \
                    --dbgaim
                  
    }


The directory into which events are written is reported.


2. have a look at the saved event
---------------------------------------

Large pos/time values
~~~~~~~~~~~~~~~~~~~~~~~~~

Very large world volume position/time values suggests targetting is missing geometry::

    OKTest --compute --save --dbgaim

    cd /tmp/blyth/opticks/OKTest/evt/g4live/torch/1

    In [1]: ox = np.load("ox.npy")   

    In [7]: ox.shape                                                                                                                                                       
    Out[7]: (10000, 4, 4)

    In [11]: ox[:,0]                                                                                                                                 
    Out[11]: 
    array([[-1531340.5  ,   407821.53 ,  -737101.44 ,     7032.428],
           [  626603.5  ,  -459454.1  , -2400000.   ,    12684.859],
           [ -952960.6  , -1668494.9  ,   652445.5  ,    16838.74 ],
           ...,
           [  864573.75 ,   -31689.084,   -56326.715,     4036.174],
           [-1001047.94 ,  -657957.5  ,   578423.4  ,    20682.805],
           [ -697387.94 ,  1256682.9  , -2400000.   ,     9331.314]], dtype=float32)


Using pyvista visualize where the photons are ending up::

    In [1]: ox = np.load("ox.npy")                                                                                                                                         

    In [2]: import pyvista as pv                                                                                                                                           

    In [3]: pl = pv.Plotter()                                                                                                                                              

    In [5]: pl.add_points(ox[:,0,:3] )                                                                                                                                     
    Out[5]: (vtkRenderingOpenGL2Python.vtkOpenGLActor)0x16331fec0

    In [6]: pl.show()                                                 

Visualization shows a box of points with clumping in the middle and collection on sides. 
This suggests are targetting node 0 world volume which misses all geometry.

Quick way to do this::

     cd /tmp/blyth/opticks/OKTest/evt/g4live/torch/1
     vtk_oxplt.py 


Small or unspread pos/time values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Very small position time values suggests are getting stuck in some geometry::

   cd /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe846377973af10a/1/tmp/blyth/OKTest/evt/g4live/torch/1

   In [1]: ox = np.load("ox.npy")

    In [3]: ox.shape
    Out[3]: (10000, 4, 4)

    In [4]: ox[:,0,0]
    Out[4]: 
    array([-0.7750918, -4.082367 ,  1.0890563, ...,  3.6566215,  3.7688375,
            1.055684 ], dtype=float32)

    In [5]: ox[:,0]                 ## these are positions of final photons in mm and time in ns, are not getting far
    Out[5]: 
    array([[-0.7750918 ,  0.04091384, -4.        ,  0.23885815],
           [-4.082367  ,  1.0382159 , -1.4258773 ,  0.12604423],
           [ 1.0890563 ,  4.085922  , -0.27708182,  0.27266118],
           ...,
           [ 3.6566215 , -3.7540717 ,  2.503454  ,  0.11937293],
           [ 3.7688375 ,  3.445465  , -0.71644664,  0.2556168 ],
           [ 1.055684  ,  4.083588  , -3.141241  ,  0.24330705]],
          dtype=float32)



Adjusting timemax and animtimemax over serveral launches allows to see that 
the small boxes (placerholder for guidetube torus) are impeding the torch "calibration" source.

TODO: make animtimemax and timemax adjustable interactively ?

Hmm changing geometry is a bit difficult, how to just nudge the source position by eg 10cm in +Z ?

::

    2019-05-09 15:30:15.292 INFO  [67331] [OpticksHub::loadGeometry@524] ]
    2019-05-09 15:30:15.292 ERROR [67331] [OpticksGen::makeLegacyGensteps@195]  code 4096 srctype TORCH
    2019-05-09 15:30:15.292 INFO  [67331] [Opticks::makeSimpleTorchStep@2215] Opticks::makeSimpleTorchStep config  cfg NULL
    2019-05-09 15:30:15.292 ERROR [67331] [OpticksGen::makeTorchstep@374]  as torchstep isDefault replacing placeholder frame  frameIdx : 0 detectorDefaultFrame : 0
    2019-05-09 15:30:15.292 INFO  [67331] [OpticksGen::targetGenstep@304] OpticksGen::targetGenstep setting frame 0 1.0000,0.0000,0.0000,0.0000 0.0000,1.0000,0.0000,0.0000 0.0000,0.0000,1.0000,0.0000 0.0000,0.0000,0.0000,1.0000
    2019-05-09 15:30:15.293 INFO  [67331] [SLog::SLog@12]  ( OpticksViz::OpticksViz 


::

    2209 TorchStepNPY* Opticks::makeSimpleTorchStep()
    2210 {
    2211     const std::string& config = m_cfg->getTorchConfig() ;
    2212 
    2213     const char* cfg = config.empty() ? NULL : config.c_str() ;
    2214 
    2215     LOG(info) 
    2216               << " enable : --torch (the default) "
    2217               << " configure : --torchconfig [" << ( cfg ? cfg : "NULL" ) << "]"
    2218               << " dump details : --torchdbg "            
    2219               ;
    2220     
    2221     TorchStepNPY* torchstep = new TorchStepNPY(TORCH, 1, cfg );
    2222     
    2223     unsigned int photons_per_g4event = m_cfg->getNumPhotonsPerG4Event() ;  // only used for cfg4-
    2224     
    2225     torchstep->setNumPhotonsPerG4Event(photons_per_g4event);
    2226     
    2227     return torchstep ;
    2228 }   
    2229 


::

    blyth@localhost optickscore]$ opticks-f makeSimpleTorchStep 
    ./opticksgeo/OpticksGen.cc:    TorchStepNPY* torchstep = m_ok->makeSimpleTorchStep();
    ./optickscore/Opticks.hh:       TorchStepNPY*        makeSimpleTorchStep();
    ./optickscore/Opticks.cc:TorchStepNPY* Opticks::makeSimpleTorchStep()



::

    366 TorchStepNPY* OpticksGen::makeTorchstep()
    367 {
    368     TorchStepNPY* torchstep = m_ok->makeSimpleTorchStep();
    369 
    370     if(torchstep->isDefault())
    371     {
    372         int frameIdx = torchstep->getFrameIndex();
    373         int detectorDefaultFrame = m_ok->getDefaultFrame() ;
    374         LOG(error)
    375             << " as torchstep isDefault replacing placeholder frame "
    376             << " frameIdx : " << frameIdx
    377             << " detectorDefaultFrame : " << detectorDefaultFrame
    378             ;
    379 
    380         torchstep->setFrame(detectorDefaultFrame);
    381     }
    382 
    383 
    384     targetGenstep(torchstep);  // sets frame transform
    385     setMaterialLine(torchstep);
    386     torchstep->addActionControl(OpticksActionControl::Parse("GS_TORCH"));
    387 
    388     bool torchdbg = m_ok->hasOpt("torchdbg");
    389     torchstep->addStep(torchdbg);  // copyies above configured step settings into the NPY and increments the step index, ready for configuring the next step 
    390 
    391     NPY<float>* gs = torchstep->getNPY();
    392     gs->setArrayContentVersion(-OPTICKS_VERSION_NUMBER) ;
    393 
    394     if(torchdbg) gs->save("$TMP/torchdbg.npy");
    395 
    396     return torchstep ;
    397 }


::

    1093 int Opticks::getDefaultFrame() const
    1094 {
    1095     return m_resource->getDefaultFrame() ;
    1096 }

    0377 int OpticksResource::getDefaultFrame() const
     378 {
     379     return m_default_frame ;
     380 }


     66 const int BOpticksResource::DEFAULT_FRAME_OTHER = 0 ;
     67 const int BOpticksResource::DEFAULT_FRAME_DYB = 3153 ;
     68 const int BOpticksResource::DEFAULT_FRAME_JUNO = 62593 ;
     69 


Hmm looks like live geometry is not identified as JUNO so uses DEFAULT_FRAME_OTHER 0. 



Actually better to change the geometry 
-------------------------------------------

* :doc:`torus_replacement_on_the_fly`

::

    247 geocache-j1808-v4()
    248 {
    249     local iwd=$PWD
    250     local tmp=$(geocache-tmp $FUNCNAME)
    251     mkdir -p $tmp && cd $tmp
    252 
    253     type $FUNCNAME
    254     opticksdata-
    255 
    256     gdb --args \
    257     OKX4Test --gdmlpath $(opticksdata-jv4) --csgskiplv 22
    258 
    259     cd $iwd
    260 }


* hmm an opportunity to get memory profiling working, as this is the memory intensive translation
  that fails on lxslc

  * :doc:`geocache-j1808-v3-bad-alloc-late-on`



