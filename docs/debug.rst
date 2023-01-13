Debug
======

Debugging optical photon propagation using NumPy + ipython
-------------------------------------------------------------

Using *num_hits* to debug an optical propagation is hopeless.

You need to enable photon step-by-step recording and 
save the corresponding arrays in NumPy .npy files.
Then you can examine the parameters of all the photons including
history flags at every step point of their propagation
(up to a configured maximum number of step points) from
interactive ipython sessions.::

    ipython> a = np.load("/path/to/photon.npy")

You could then make plots drawing the paths of the photons.
I recommend pyvista if your want to do that
A convenient way to install pyvista is to use anaconda.
The more commonly used matplotlib python plotting library
is not good with 3D plotting or large data sets.

To save the arrays you need to::

    export OPTICKS_EVENT_MODE=StandardFullDebug   # configure step point recording
    export G4CXOpticks__simulate_saveEvent=1      # enable saveEvent from G4CXOpticks::simulate

    # optionally enable logging in relevant classes
    export G4CXOpticks=INFO
    export SEventConfig=INFO


What OPICKS_EVENT_MODE does
-------------------------------

To see how OPTICKS_EVENT_MODE works look at::

    sysrap/SEventConfig.hh
    sysrap/SEventConfig.cc

Especially::

    324 int SEventConfig::Initialize() // static
    325 {
    326     const char* mode = EventMode();
    327     int maxbounce = MaxBounce();
    328
    329     if(strcmp(mode, Default) == 0 )
    330     {
    331         SetCompMaskAuto() ;   // comp set based on Max values
    332     }
    333     else if(strcmp(mode, StandardFullDebug) == 0 )
    334     {
    335         SEventConfig::SetMaxRecord(maxbounce+1);
    336         SEventConfig::SetMaxRec(maxbounce+1);
    337         SEventConfig::SetMaxSeq(maxbounce+1);
    338         SEventConfig::SetMaxPrd(maxbounce+1);
    339
    340         // since moved to compound sflat/stag so MaxFlat/MaxTag should now either be 0 or 1, nothing else
    341         SEventConfig::SetMaxTag(1);
    342         SEventConfig::SetMaxFlat(1);
    343         SetCompMaskAuto() ;   // comp set based on Max values
    344     }
    345     else
    346     {
    347         LOG(fatal) << "mode [" << mode << "] IS NOT RECOGNIZED "  ;
    348         LOG(fatal) << " options : " << Default << "," << StandardFullDebug ;
    349         assert(0);
    350     }



Possible arrays saved by sysrap/SEvt
----------------------------------------

The default is not saving any arrays.

Array saving is regarded as purely a debugging activity
that should not be done in production, as it will greatly slow down 
the simulation and write many large files. 

Configure what arrays to save and their limits with *SEventConfig* 
via envvars or static methods. 


+------------+------------------+-------------------------------------------------------------------------------+
|  array     |  shape           |  notes                                                                        |
+============+==================+===============================================================================+
| *genstep*  | (num_gs,6,4)     |  parameters of Cerenkov/Scintillation/Torch/... generation                    |
+------------+------------------+-------------------------------------------------------------------------------+
| *photon*   | (num_ph,4,4)     |  sysrap/sphoton.h : final photon params : position, time, mom, pol, flags     |
+------------+------------------+-------------------------------------------------------------------------------+
| *record*   | (num_ph,16,4,4)  |  sysrap/sphoton.h : step point records (configurable num points)              | 
+------------+------------------+-------------------------------------------------------------------------------+
| *hit*      | (num_ht,4,4)     |  sysrap/sphoton.h : selection of *photon*                                     |
+------------+------------------+-------------------------------------------------------------------------------+
| *rec*      | (num_ph,16,4,4)  |  compressed rec (no longer used, use *record* instead)                        |                     
+------------+------------------+-------------------------------------------------------------------------------+
| *seq*      | (num_ph,2)       |  sysrap/sseq.h photon level step-by-step history and material recording       |
+------------+------------------+-------------------------------------------------------------------------------+
| *prd*      |                  |                                                                               |
+------------+------------------+-------------------------------------------------------------------------------+


SEventConfigTest
------------------

The **SEventConfigTest** binary can be used to dump 
the configuration.  

::

    epsilon:sysrap blyth$ SEventConfigTest 
    test_EstimateAlloc@20: 
    SEventConfig::Desc
           OPTICKS_EVENT_MODE          EventMode  : Default
         OPTICKS_RUNNING_MODE        RunningMode  : 0
                                RunningModeLabel  : SRM_DEFAULT
         OPTICKS_G4STATE_SPEC        G4StateSpec  : 1000:38
                                G4StateSpecNotes  :  38=2*17+4 is appropriate for MixMaxRng  
        OPTICKS_G4STATE_RERUN       G4StateRerun  : -1
          OPTICKS_MAX_GENSTEP         MaxGenstep  : 1000000
           OPTICKS_MAX_PHOTON          MaxPhoton  : 1000000
         OPTICKS_MAX_SIMTRACE        MaxSimtrace  : 1000000     MaxCurandState  : 1000000
           OPTICKS_MAX_BOUNCE          MaxBounce  : 9
           OPTICKS_MAX_RECORD          MaxRecord  : 0
              OPTICKS_MAX_REC             MaxRec  : 0
              OPTICKS_MAX_SEQ             MaxSeq  : 0
              OPTICKS_MAX_PRD             MaxPrd  : 0
              OPTICKS_MAX_TAG             MaxTag  : 0
             OPTICKS_MAX_FLAT            MaxFlat  : 0
             OPTICKS_HIT_MASK            HitMask  : 64
                                    HitMaskLabel  : SD
           OPTICKS_MAX_EXTENT          MaxExtent  : 1000
             OPTICKS_MAX_TIME            MaxTime  : 10
              OPTICKS_RG_MODE             RGMode  : 2
                                     RGModeLabel  : simulate
            OPTICKS_COMP_MASK           CompMask  : 262
                                   CompMaskLabel  : genstep,photon,hit
             OPTICKS_OUT_FOLD            OutFold  : $DefaultOutputDir
             OPTICKS_OUT_NAME            OutName  : -
    OPTICKS_PROPAGATE_EPSILON   PropagateEpsilon  :     0.0500
         OPTICKS_INPUT_PHOTON        InputPhoton  : -

    test_EstimateAlloc@25: al.desc
    -
    epsilon:sysrap blyth$ 



Changing the OPTICKS_EVENT_MODE envvar to "StandardFullDebug" has a large effect on the config::

    epsilon:sysrap blyth$ OPTICKS_EVENT_MODE=StandardFullDebug SEventConfigTest  
    test_EstimateAlloc@20: 
    SEventConfig::Desc
           OPTICKS_EVENT_MODE          EventMode  : StandardFullDebug
         OPTICKS_RUNNING_MODE        RunningMode  : 0
                                RunningModeLabel  : SRM_DEFAULT
         OPTICKS_G4STATE_SPEC        G4StateSpec  : 1000:38
                                G4StateSpecNotes  :  38=2*17+4 is appropriate for MixMaxRng  
        OPTICKS_G4STATE_RERUN       G4StateRerun  : -1
          OPTICKS_MAX_GENSTEP         MaxGenstep  : 1000000
           OPTICKS_MAX_PHOTON          MaxPhoton  : 1000000
         OPTICKS_MAX_SIMTRACE        MaxSimtrace  : 1000000     MaxCurandState  : 1000000
           OPTICKS_MAX_BOUNCE          MaxBounce  : 9
           OPTICKS_MAX_RECORD          MaxRecord  : 10
              OPTICKS_MAX_REC             MaxRec  : 10
              OPTICKS_MAX_SEQ             MaxSeq  : 10
              OPTICKS_MAX_PRD             MaxPrd  : 10
              OPTICKS_MAX_TAG             MaxTag  : 1
             OPTICKS_MAX_FLAT            MaxFlat  : 1
             OPTICKS_HIT_MASK            HitMask  : 64
                                    HitMaskLabel  : SD
           OPTICKS_MAX_EXTENT          MaxExtent  : 1000
             OPTICKS_MAX_TIME            MaxTime  : 10
              OPTICKS_RG_MODE             RGMode  : 2
                                     RGModeLabel  : simulate
            OPTICKS_COMP_MASK           CompMask  : 274814
                                   CompMaskLabel  : genstep,photon,record,rec,seq,prd,hit,tag,flat,aux
             OPTICKS_OUT_FOLD            OutFold  : $DefaultOutputDir
             OPTICKS_OUT_NAME            OutName  : -
    OPTICKS_PROPAGATE_EPSILON   PropagateEpsilon  :     0.0500
         OPTICKS_INPUT_PHOTON        InputPhoton  : -

    test_EstimateAlloc@25: al.desc
    -
    epsilon:sysrap blyth$ 





Saving Photon Propagations into NumPy arrays
----------------------------------------------

To see what G4CXOpticks__simulate_saveEvent is doing look at g4cx/G4CXOpticks.cc simulate method. 


The directory where the numpy arrays is saved is based
on your executable name and the event index you set with::

   SEvt::SetIndex(eventid);

Enable logging in SEvt to see what it is::

    export SEvt=INFO

Opticks has lots of python machinery for loading and presenting
such NumPy .npy arrays in the "ana" directory and all over the place.
However it is better to examine them manually using ipython to begin with,
because most people will need to improve their NumPy+ipython skills
to make best use of this debugging info and to be able to understand
the python machinery.



Debugging Lack of Hits
---------------------------


When none of your photons have flagmask SURFACE_DETECT you will get no hits.
You will still have gensteps, photons and records without having hits.

If your photon propagation histories are as expected and you
are still not getting hits then your problem is probably that the geometry
translation did not notice your sensitive detectors somehow OR
that you do not have any sensitive detectors.

A common cause of this is loading a geometry from GDML and not
reinstating the association.


*hits* are the subset of *photons* with flagmask matching the hitmask (default SURFACE_DETECT)
so when you get no hits it means that none of your photons .flagmask has
all the bits of the hitmask set.

You can of course select the hits array from the photons array using one line of NumPy,
but that will just match with NumPy what the C++/CUDA would do.

You can learn about the mechanics of hit selection in::

   ~/opticks/notes/mechanics_of_hit_selection.rst
   https://bitbucket.org/simoncblyth/opticks/src/master/notes/mechanics_of_hit_selection.rst



Checking photon propagation histories
----------------------------------------

So in ipython::

    import numpy as np 
    r = np.load("/tmp/ami/opticks/GEOM/example_pet/ALL/007/record.npy”)

    r[:,:5,0]   # (pos, time) of first 5 step point of all photons
    r[0, :5, 0]   # (pos, time) of first 5 step points of the first photon (slot 0)


So to start debugging I would look at the sequence of positions, 
times, momentum directions and polarizations and step point flags to 
see if they are what I would expect. 



~/opticks/ana/tests/check.sh : setup environment for NumPy debug
--------------------------------------------------------------------

::

    #!/bin/bash -l 

    usage(){ cat << EOU
    check.sh
    ==========

    Setup environment:

    PYTHONPATH=$HOME
       allows python scripts to import opticks python machinery 
       eg with  "from opticks.ana.fold import Fold"

    CFBASE=$HOME/.opticks/GEOM/J004
       configures where to load geometry from

    FOLD=$CFBASE/G4CXSimulateTest/ALL
       configures the directory to load event arrays from, 
       the directory is up to the user

    To a large degree the directory positions of geometry 
    and event files are controlled by the user. 
    However the example of versioning a geometry name "J004"
    and keeping event folders within the corresponding 
    geometry folder is a good one to follow as it is important 
    to retain the connection between event data and the geometry used
    to create the event data.  

    EOU
    }

    export PYTHONPATH=$HOME  
    export CFBASE=$HOME/.opticks/GEOM/J004
    export FOLD=$CFBASE/G4CXSimulateTest/ALL

    ${IPYTHON:-ipython} --pdb -i check.py 


~/opticks/ana/tests/check.py : python basic example
------------------------------------------------------

::

    epsilon:tests blyth$ cat check.py 
    #!/usr/bin/env python

    import numpy as np
    from opticks.ana.fold import Fold
    from opticks.ana.p import *
    from opticks.ana.histype import HisType

    if __name__ == '__main__':
        f = Fold.Load(symbol="f")
        print(repr(f))

        p = f.photon
        r = f.record
        s = f.seq
        h = f.hit 



Debugging Geometry
----------------------

To debug geometry issues you need to have some
familiarity with the translation. It starts from G4CXOpticks::setGeometry.

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


X4Geo::Translate
   old way with loads of code, entire extg4 package

U4Tree::Create
   is a simpler approach to translation that I am starting to develop
   which is aiming to go directly

The translation code is still in flux with both old and new
approaches in use and an entire geometry model too many.::

    .       extg4         CSG_GGeo
    Geant4  ---->   GGeo ------->   CSG


The CSG_GGeo package translates the GGeo geometry model
into CSG which gets upload to GPU.

While I dont suggest you try to understand
the geometry translation in detail, you need some familiarity
with how the sensitive detectors get translated in order to
debug your issue.

The best way to start debugging geometry is to persist it by rerunning with::

    export G4CXOpticks=INFO
    export G4CXOpticks__setGeometry_saveGeometry=$HOME/.opticks/GEOM/$GEOM

The above envvar configures directory to save the geometry.

Then you can run small executables/scripts
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



