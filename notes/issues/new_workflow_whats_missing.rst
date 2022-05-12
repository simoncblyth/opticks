new_workflow_whats_missing
=============================

Most items have been done before in the old workflow, so 
just needs to be revisited and brought over to new environment. 


Generation
--------------

1. cerenkov generation : needs integrating and testing 

   * DONE : integrated the standard rejection sampling approach in QCerenkov/qcerenkov

   * complicated by float precision rejection sampling giving 
     poor wavelength match
   * technique enabling float precision wavelength matching requires 
     preparation of icdf for all materials, currently tested only with LS
   * plus need machinery to use the appropriate icdf on device 


   * TODO: CerenkovStandalone mocking Geant4 to do what QSimTest cerenkov_generate 
     does with from the same input QDebug cerenkov_gs : then random aligned comparison ?

     * HMM: easier to do scint first as lots of history of working on similar with Cerenkov 


2. scintillation generation, reemission : needs integrating


   * expect straightforward as wavelength matched OK previously
     using float precision  

   * DONE : integrated standard lookup approach in QScint/qscint 

   * TODO: focussed validation ScintillationStandalone mocking Geant4 to do what QSimTest scint_generate 
     does from the same input QDebug scint_gs  : then random aligned comparison ?

     * this requies hacking geant4 scintillation generation loop to accept gensteps as input
       "jcv DsG4Scintillation" BUT without changing the code too much, 
       the point is comparison afterall 

     * note this is easier than BoundaryStandalone because it is only the 
       generation loop part of DsG4Scintillation::PostStepDoIt that gets done 
       on GPU so it is only that which needs comparison with the GPU implementation

       * ScintGenStandalone is more appropriate name 
       * ScintillationIntegral is the most involved part, done in qscint.h with scint_tex  
       * HMM: recall doing something like this before, but cannot find it. 



Pass the parcel ? Maybe use an SSim to hold the simulation input arrays::

     * CSGFoundry that is focussed on geometry holding the icdf feels wrong 

     200 void CSG_GGeo_Convert::convertScintillatorLib()
     201 {
     202     GScintillatorLib* slib = ggeo->getScintillatorLib();
     203     NP* icdf = slib->getBuf();   // assuming 1 scintillator
     204     foundry->icdf = icdf ;
     205 }






Approach : scerenkov.h sscintillation.h
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow the pattern established in:: 

     sysrap/storch.h
     sysrap/tests/storch_test.cc
     sysrap/tests/storch_test.py
     sysrap/tests/storch_test.sh  


Integrate into qsim::

    1980 template <typename T>
    1981 inline QSIM_METHOD void qsim<T>::generate_photon(sphoton& p, curandStateXORWOW& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const
    1982 {
    1983     quad4& q = (quad4&)p ;
    1984     const int& gencode = gs.q0.i.x ;
    1985 
    1986     switch(gencode)
    1987     {
    1988         case OpticksGenstep_PHOTON_CARRIER:  generate_photon_carrier(q, rng, gs, photon_id, genstep_id)  ; break ;
    1989         case OpticksGenstep_TORCH:           storch::generate(       p, rng, gs, photon_id, genstep_id ) ; break ;
    1990         default:                             generate_photon_dummy(  q, rng, gs, photon_id, genstep_id)  ; break ;
    1991     }
    1992 }






Engine Change
----------------

3. change G4Opticks "engine" to use CSGOptiX/CSGOptiX 

   * interface for geometry, gensteps, hits is straightforward, 
     but likely to require changes to CSGOptiX, CSG_GGeo

   * also needs some development to improve flexibility of handling 
     of varying simulation physics input arrays, will start 
     by trying to work with a std::map<std::string, NP*> 
     or a directory containing various arrays

     * MultiFilm texture arrays
     * cerenkov icdf
     * scintillation icdf
     * boundary properties


New Event Handling
-----------------------

* :doc:`event_handling_into_new_workflow`

4. okc/OpticksEvent replaced by qudarap/QEvent

   * machinery for Opticks vs Geant4 comparison
   * Geant4 CFG4/CRecorder needs reworking to to write QEvent (plan U4 pkg to do this) 
   * python analysis comparison machinery needs update

5. GPU launch during event genstep collection (not just at end of event)

   * GPU launches should happen once a configured number of photons is reached
   * better suited to fixed+reused QEvent photon buffers


Identity Mechanics for PMT efficiency, angular efficiency, MultiFilm
------------------------------------------------------------------------


6. identity machinery, instance level and shape/boundary level, needed for:  

   * PMT efficiency
   * PMT type for MultiFilm 

7. PMT angular efficiency for on device efficiency culling 




