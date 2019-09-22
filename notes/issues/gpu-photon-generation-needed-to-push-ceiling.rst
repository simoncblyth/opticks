gpu-photon-generation-needed-to-push-ceiling
================================================

issue
-------

* want to push the limits of how many photons, current max is 100M
* scanning high photon counts is slow because of having to generate
  loadsa input photons on the gpu : this was done for alignment. 
* when not doing G4/alignment can return to GPU photon generation, but how ? 

  * emitter stuff in the geometry ?

::

   emitconfig = "photons:100000,wavelength:380,time:0.0,posdelta:0.1,sheetmask:0x1,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55" 



how to proceed
------------------

* get some old tboolean which do not use emitconfig running 
* find the old way of configuring photon sources

::

   tboolean-interlocked --nog4



* see documentation in tboolean : Configuring Photon Sources 
* looks like torchconfig still being passed as argument, 
  but is ignored when an emitter is found in the geometry 


::

    [blyth@localhost ana]$ tboolean-
    [blyth@localhost ana]$ tboolean-torchconfig
    type=disc_photons=100000_mode=fixpol_polarization=1,1,0_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000_source=0,0,599_target=0,0,0_time=0.1_radius=300_distance=200_zenithazimuth=0,1,0,1_material=Vacuum_wavelength=500



* TODO: make torchconfig honour generateoverride


::

    torchconfig
    ~~~~~~~~~~~

    opticksnpy/TorchStepNPY 
        parses config string and encodes params into genstep buffer for copying to GPU 

    optixrap/cu/torchstep.h
        GPU side OptiX generation of photons from the genstep buffer, this 
        works by throwing two random numbers (ranges from zeaz:zenithazimuth)
        that are used in different ways based upon genstep params  

    cfg4/CTorchSource
        CPU side G4 generation of photons from the genstep buffer, actually the TorchStepNPY instance


    Deficiencies of torchconfig:

    1. get bizarre results when the torch positions are outside the container that 
       defines the Opticks domain, forcing manually tweaking of the 
       torch positions for different containers : the problem with this
       is that it is then not possible to reproduce a prior torch setup via a 
       torchname for example. 

       *emitconfig* solves this issue by decoupling from position, but 
       is currently limited to all sheet/face generation.




Enabling generateoverride to work with torchconfig
------------------------------------------------------

::

    197 void GenstepNPY::setNumPhotons(unsigned int num_photons)
    198 {
    199     m_ctrl.w = num_photons ;
    200 }



* did this in OpticksGen::makeTorchstep

::

    tboolean-;tboolean-interlocked --nog4 --generateoverride 100 
    tboolean-;tboolean-interlocked --nog4 --generateoverride -10 --rngmax 10
        ## visualizing 10M photons works OK on TITAN RTX



