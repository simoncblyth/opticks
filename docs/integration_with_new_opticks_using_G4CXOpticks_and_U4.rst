integration_with_new_opticks_using_G4CXOpticks_and_U4
========================================================

Overview
----------

This document aims to assist people wanting to integrate 
the new Opticks workflow with Geant4 applications. 

While there is currently a lack of full examples for
doing such and integration there unit tests 
of all classes/structs.   


Interface
----------

The below are the main interface classes in new Opticks.
You are suggested to study these classes::


g4cx/G4CXOpticks.hh 
   translate the geometry, launch simulation, get hits 

u4/U4.hh
   collect gensteps into sysrap/SEvt.hh, 

sysrap/SEventConfig.hh
   configures components and structure of sysrap/SEvt arrays
   based on envvars or calls to static methods 
   that are make **prior to SEvt instanciation** 
   which is done by G4CXOpticks::setGeometry


Translation of Geometry
-------------------------

Translate the geometry::

    #include "G4CXOpticks.hh" 
    if( opticksMode != 0 ) G4CXOpticks::SetGeometry(world) ;


Collection of Gensteps
-------------------------

Opticks works by moving the optical photon generation
and propagation to the GPU. It knows what to generate using
the parameters of the gensteps which you must collect.

The Geant4 processes that generate optical photons
which are relevant to your detector 
(eg for JUNO thats G4Scintillation and G4Cerenkov) 
need to be modified in order to collect gensteps (using u4/U4.hh)

Gensteps are collected just prior to the optical photon generation loop
in the PostStepDoIt method of the processes
and the generation loop is skipped in order to get speedups.


In each optical photon generating process PostStepDoIt before the generation loop 
collecting gensteps with u4/U4.hh::

    U4::CollectGenstep_DsG4Scintillation_r4695
    U4::CollectGenstep_G4Cerenkov_modified

For debugging its also useful to do both Geant4 and Opticks 
propagations, when doing that setup photon labels on the secondaryTrack 
with the below methods that have machinery to follow photons thru 
reemission generations::

    U4::GenPhotonAncestor
    U4::GenPhotonBegin
    U4::GenPhotonEnd

So in Scintillation and Cerenkov PostStepDoIt::

      #include "U4.hh"

      U4::GenPhotonAncestor(&aTrack);

      if(NumPhoton > 0 && (m_opticksMode > 0))
      {
          U4::CollectGenstep_DsG4Scintillation_r4695( &aTrack, &aStep, NumPhoton, scnt, ScintillationTime);

          OR

          U4::CollectGenstep_G4Cerenkov_modified(
              &aTrack,
              &aStep,
              fNumPhotons,
              BetaInverse,
              Pmin,
              Pmax,
              maxCos,
              maxSin2,
              MeanNumberOfPhotons1,
              MeanNumberOfPhotons2
          );

      }



        if(m_opticksMode != 1)  // still do CPU generation loop, eg for opticksMode 3
         {

             for(G4int i = 0 ; i < NumPhoton ; i++)
             {
                  U4::GenPhotonBegin(i);


                  // standard Geant4 photon generation


                  U4::GenPhotonEnd(i, aSecondaryTrack);
              }
          }



Find examples::

    epsilon:opticks blyth$ grep U4::Collect u4/tests/*.cc
    u4/tests/Local_DsG4Scintillation.cc:            U4::CollectGenstep_DsG4Scintillation_r4695( &aTrack, &aStep, NumPhoton, scnt, ScintillationTime);
    u4/tests/Local_G4Cerenkov_modified.cc:  U4::CollectGenstep_G4Cerenkov_modified(



Launch GPU Simulation, generating and propagating sphoton.h on GPU, gather configured components
---------------------------------------------------------------------------------------------------

::

    #include "G4CXOpticks.hh"
    #include "SEvt.hh"
    #include "U4Hit.h"
    #include "U4HitGet.h"
    #include "U4Recorder.hh"


    176 #ifdef WITH_G4CXOPTICKS
    179     SEvt::SetIndex(eventID);

    182     unsigned num_genstep = SEvt::GetNumGenstepFromGenstep();
    183     unsigned num_photon  = SEvt::GetNumPhotonCollected();
    184
    185     G4CXOpticks::Get()->simulate() ;
    186
    187     unsigned num_hit = SEvt::GetNumHit() ;
    189
    190     LOG(LEVEL)
    191         << " eventID " << eventID
    192         << " num_genstep " << num_genstep
    193         << " num_photon " << num_photon
    194         << " num_hit " << num_hit
    196         ;
    197


Hit Handling
--------------

Note that the hits are merely a subset of the photons 
that are represented by sysrap/sphoton.h struct on CPU and GPU.

U4Hit provides an example to convert those into Geant4 types.
However, you do not need to use U4Hit, it is just an example. 

You can easily create your own converters to translate from the 
underlying sphoton.h into the hit type needed by your simulation application. 
If you have lots of hits this can avoid pointless conversions through 
multiple types. 

::

    ...
    235 #ifdef WITH_G4CXOPTICKS
    236     U4Hit hit ;
    237     U4HitExtra hit_extra ;
    238     U4HitExtra* hit_extra_ptr = way_enabled ? &hit_extra : nullptr ;
    239     for(int idx=0 ; idx < int(num_hit) ; idx++)
    240     {
    241         U4HitGet::FromEvt(hit, idx);
    242         collectHit(&hit, hit_extra_ptr, merged_count, savehit_count );
    243         if(idx < 20 && LEVEL == info) ss << descHit(idx, &hit, hit_extra_ptr ) << std::endl ;
    244     }
    245


The U4HitGet is converting from Opticks SEvt into Geant4 types within U4Hit::

     52 inline void U4HitGet::FromEvt(U4Hit& hit, unsigned idx )
     53 {
     54     sphoton global, local  ;
     55     SEvt* sev = SEvt::Get();
     56     sev->getHit( global, idx);
     57
     58     sphit ht ;
     59     sev->getLocalHit( ht, local,  idx);
     60
     61     ConvertFromPhoton(hit, global, local, ht );
     62 }
     63




Logging Control
-----------------

Logging of almost all classes/struct can be controlled by setting 
envvars corresponding to the name of the classs/struct, eg::

    export SEvt=INFO
   


About Geometry Translation in G4CXOpticks::setGeometry
-----------------------------------------------------------

The translation code is still in flux with both old and new
approaches in use and an entire geometry model too many.::


    .        extg4         CSG_GGeo
    Geant4  ---->   GGeo ------->   CSG

The CSG_GGeo package translates the GGeo geometry model
into CSG which gets upload to GPU.

X4Geo::Translate
   old way with loads of code, entire extg4 package, still in use

U4Tree::Create
   is a simpler approach to translation that I am starting to develop
   which is aiming to go directly


::

    210 void G4CXOpticks::setGeometry(const G4VPhysicalVolume* world )
    211 {
    212     LOG(LEVEL) << " G4VPhysicalVolume world " << world ;
    213     assert(world);
    214     wd = world ;
    215
    216     sim = SSim::Create();
    217     stree* st = sim->get_tree();
    219     tr = U4Tree::Create(st, world, SensorIdentifier ) ;
    220 
    221  
    222     // GGeo creation done when starting from a gdml or live G4,  still needs Opticks instance
    223     Opticks::Configure("--gparts_transform_offset --allownokey" );
    224 
    225     GGeo* gg_ = X4Geo::Translate(wd) ;
    226     setGeometry(gg_);
    227 }


Debugging Geometry Issues with NumPy+ipython
-----------------------------------------------

The best way to start debugging geometry is to persist it by rerunning with::

    export G4CXOpticks=INFO
    export G4CXOpticks__setGeometry_saveGeometry=$HOME/.opticks/GEOM/$GEOM

The above envvar configures directory to save the geometry.

Then you can run small executables or python scripts
which load various parts of the persisted geometry and run tests.
One, of many, of such tests is sysrap/tests/stree_test.sh
Build and use that::

    cd ~/opticks/sysrap/tests

    ./stree_test.sh build
        ## builds stree_test binary

    ./stree_test.sh run
        ## these load the geometry into C++ and run tests against it
        ## one of the tests dumps sensor info

    ./stree_test.sh ana
        ## this loads the same geometry into ipython
        ## and run tests against it



How to report problems
-------------------------


Whenever you get asserts please run under gdb and provide a backtrace.
The backtrace gives precisely the call stack that resulted in the assert.

Collect a backtrace using gdb::

    gdb /path/to/execuable
    r   # run

    bt  # short for backtrace, after hit assert



Making Your GPU Geometry Faster
---------------------------------
    
The translation gets exercised mostly with highly factorizable geometry
with many thousands of PMTs that become instanced.
Instancing greatly reduces GPU memory resources for geometry
that has many repeated elements.
    
This factorization works by computing a digest
(based on all the transforms and shape indices of the subtree)
for every subtree of the entire tree of volumes.  
Then repeated subtrees are identified as "factored" pieces of the
geometry that get instanced : ie treated as identical just requiring
a different transform to place them.

The volumes with subtrees that are not repeated enough times 
to pass instancing cuts are treated as "remainder" volumes  
(the cut is something like 500 repeats).
All the remainder volumes are treated together in the so
called global factor (with repeat index 0) which does not
get instanced.

There is potentially a large performance differences between
an instanced geometry and an all global one.
But this performance difference  will be very dependent
on the geometry so its good to do both and compare performance.

