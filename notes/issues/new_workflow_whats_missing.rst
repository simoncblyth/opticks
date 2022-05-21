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



sgs : examples/Geant4/ScintGenStandalone 
---------------------------------------------

* TODO: encapsulate the details of genstep collection, with higher level API

::

      68 #ifdef STANDALONE
      69 
      70 // HMM: how to collect gensteps less disruptively in higher level way, with just the one header ?
      71 // this is baring everything ... try keeping genstep implementation behind the curtains ?
      72 
      73 #include "scuda.h"
      74 #include "squad.h"
      75 #include "sgs.h"
      76 
      77 #include "spho.h"
      78 #include "U4.hh"
      79 #include "U4PhotonInfo.h"
      80 #include "SEvt.hh"
      81 




Handling VRAM MAX_PHOTON limits ? Launch during genstep collection or split launches at end of event, or across multiple events 
---------------------------------------------------------------------------------------------------------------------------------

Suspect its simpler to collect all gensteps and then do split launches at end
of event if the total photon count exceeds configured limits. 
Doing the launch during collection means you do not know the future 
gensteps to be collected so you cannot arrange splits optimally. 
This is especially the case when there are multiple GPUs with different VRAM
and hence different photon limits. 

Given that the simulation of every genstep is independent of all others, 
optical photon simulation on multiple GPUs should be able to use the simplest 
of multi-GPU techniques. Thats got to be good for development time and performance. 

* https://medium.com/gpgpu/multi-gpu-programming-6768eeb42e2c

The above has a simple example of using multiple CPU threads where each CPU thread 
talks with a different GPU cudaSetDevice. 

It would be simplest to duplicate the geometry buffers and textures on
all GPUs : only if those were very large would it be worth trying to share them.



DONE : SSim, input array management
---------------------------------------
:w

1. avoids setup duplication and passing the parcel 
2. establishes more separation of concerns between CSGFoundry and SSim 
   whilst still persisting SSim into a subfolder of CSGFoundry 
3. moved SSim population into GGeo::convertSim, from CSG_GGeo_Convert::convertSim
 



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


* WIP : "gx" g4cx 



G4Opticks Into New workflow ?
--------------------------------

* event handling is near fully reimplemented in QUDARap/QSim/QEvent 
  (replacing that part of OptiXRap etc..)
 
* geometry handling needs work to bring across 

  * GGeo* G4Opticks::translateGeometry( const G4VPhysicalVolume* top )
  * TODO: can most of this be moved down to GGeo or extg4 statics ? 
  

New Workflow Top Level Package : g4cx
---------------------------------------

g4ok/G4Opticks is too involved and the dependencies too different between workflows
to change it from inside, easier to make a new pkg+class (g4cx/G4CXOpticks) 
that duplicates the important parts of the old API but is built upon the 
new workflow components:

This means any necessary functionality from old G4Opticks
that needs to be used in new workflow should if possible 
be shifted downwards to (eg down to extg4, ggeo) to both simplify 
the old G4Opticks and enable reuse in the new workflow.  


CSGOptiX 
    needs CSGFoundry geometry, sim (eg NP gensteps) passed thru to QEvent  
QUDARap
    QSim, QEvent 


CSG_GGeo
    GGeo->CSGFoundry translation 

    CSGFoundry* fd0 = CSG_GGeo_Convert::Translate(ggeo);

extg4
     Geant4->GGeo translation 

     * this brings GGeo, OpticksCore, NPY, BRAP deps 



Future Direct Geometry Workflow
----------------------------------

Currently the geometry chain is long:

* Geant4 -> x4 (NPY,GGeo) -> GGeo -> CSGFoundry 

It would be perfectly possible to do this much more directly.   
But it is significant work.  

* SO DEFER UNTIL NEW WORKFLOW OPERATIONAL

* this means the initial new workflow top package
  will have to be a franken-package combining dependencies 
  from old and new worlds

  * simulation, event handing mostly fully reimplemnented
  * geometry mostly using old workflow 


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




