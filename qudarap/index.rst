QUDARap
==========

Opticks GPU Context Prototyping
----------------------------------

qsim.h QSim.hh
    GPU context and CPU counterpart that preps it 
    acting as coordinator of all the below

QRng.hh
    loading+uploading curandState : use curand without the stack cost of curand_init

QTex.hh
    2D texture creation 

QBnd.hh
    ggeo/GBndLib -> QTex "boundary texture"

    * TODO: qbnd.h encapsulation

QScint.hh
    ggeo/GScintillatorLib -> QTex "scintillation inverse-CDF texture"

qprop.h QProp.hh
    marshalling variable length paired (energy/wavelength,property) 
    into compound array, linear interpolation with binary bin search
    just like Geant4 properties 

    * alternative to boundary texture when excactly matching Geant4 
      is more important

    * TODO: accuracy/performance comparison with QBnd   

qgs.h
    union based collective Scintillation and Cerenkov genstep handling  

qcurand.h
    templated float/double specializations for uniform access to 
    curand_uniform/curand_uniform_double 

QU.hh
    utilitles : eg device<->host copies

TODO
------

QCerenkov 
   treat it like QScint now that have workable icdf  

QEvent/qevent  
   hold refs to gensteps, seeds, photons, ...

   * coordinate with OpticksEvent or lower level NP based SEvent



Observations
-----------------

The pattern of having GPU and CPU counterparts is a useful one

* do more of that to keep qsim/QSim simple by encapsulating the pieces 
  like texture handling into qtex/QTex  




Expts
--------

qpoly.h QPoly.hh
     extern "C" expt         

qscint.h
     NOT USED ANYMORE : MOVED TO COLLECTIVE HANDLING IN qgs.h TO AVOID DUPLICATION






