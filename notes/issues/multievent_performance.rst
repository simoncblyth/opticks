Multievent Performance
=======================


Observations about performance : large pinch of salt
-----------------------------------------------------------

OSX macbook pro laptop test machine and GPU (Geforce 750M)
are very different to intended production machines, so 
measurements come with a huge pinch of salt.

Due to this should apply effort to developing 
automated performance measurement machinery 
instead of attempting to optimize on the non production system.

That said its tempting to check

* impact of move to OptiX 400, by making mesaurements with 390?
* with/without seed buffer approach


Move to USHORT (16bit) for seeds, UINT (32bit) is extravagant as just needs to hold genstep buffer index::

    m_seed_spec     = new NPYSpec(seed_     ,  0,1,1,0,      NPYBase::UINT      , "OPTIX_NON_INTEROP,OPTIX_INPUT_ONLY") ;

    USHRT_MAX   Maximum value for a variable of type unsigned short.    65535




VM growth indicates are just leaking
-------------------------------------

Need to reset/delete OpticksEvent 


0.5M tpmt timings very different now
--------------------------------------

OK 
~~~~~~~~~~~~


::
    tpmt-;tpmt-- --okg4 --compute --multievent 10    ## 0.5M photon timings look rathr 


    delta:torch blyth$ pwd
    /tmp/blyth/opticks/evt/PmtInBox/torch

    delta:torch blyth$ grep ^OPropagator::launch DeltaTime.ini
    OPropagator::launch_0=0.82421875
    OPropagator::launch_1=0.794921875
    OPropagator::launch_2=0.78515625
    OPropagator::launch_3=0.833984375
    OPropagator::launch_4=0.79296875
    OPropagator::launch_5=0.76953125
    OPropagator::launch_6=0.8046875
    OPropagator::launch_7=0.759765625
    OPropagator::launch_8=0.791015625
    OPropagator::launch_9=0.765625


Bouncemax zero (just generate) drops that thru floor::

    tpmt-;tpmt-- --okg4 --compute --multievent 2 --bouncemax 0

    delta:torch blyth$ grep ^OPropagator::launch DeltaTime.ini
    OPropagator::launch_0=0.0625
    OPropagator::launch_1=0.0234375

* so time is spent inside the bounce loop, not in setup/teardown
* perhaps skipping records will have impact now

Bouncemax 1 up to half the full bouncemax 9::

    delta:torch blyth$ grep ^OPropagator::launch DeltaTime.ini
    OPropagator::launch_0=0.404296875
    OPropagator::launch_1=0.447265625


Huh, with the default OKTest rather than OKG4Test times almost half.
Maybe host memory pressure?::

    tpmt-;tpmt--  --compute --multievent 10

    delta:torch blyth$ grep ^OPropagator::launch DeltaTime.ini
    OPropagator::launch_0=0.806640625
    OPropagator::launch_1=0.55859375
    OPropagator::launch_2=0.5
    OPropagator::launch_3=0.48828125
    OPropagator::launch_4=0.48046875
    OPropagator::launch_5=0.484375
    OPropagator::launch_6=0.478515625
    OPropagator::launch_7=0.474609375
    OPropagator::launch_8=0.466796875
    OPropagator::launch_9=0.47265625


Undefing WITH_RECORD has little effect::

    tpmt-;tpmt-- --compute --multievent 2

    delta:torch blyth$ grep ^OPropagator::launch DeltaTime.ini
    OPropagator::launch_0=0.63671875
    OPropagator::launch_1=0.453125

Putting back WITH_RECORD::

    2016-09-16 18:34:33.585 INFO  [56430] [TimesTable::dump@105] Opticks::postpropagate filter: OPropagator::launch
              7.621          0.781      30514.000          0.000 : OPropagator::launch_0
             10.637          0.641      30776.000          0.000 : OPropagator::launch_1



Revert back to 3080 by changing CMake argument and doing a major rebuild::

    delta:opticks blyth$ opticks-optix-install-dir
    /Developer/OptiX_380


TODO: arrange separate build dirs and installs for different OptiX versions
to make it less painfull to jump inbetween versions.


Performance is drastically faster with 3080, but multievent >1 is failing even with trivial::

    tpmt-;tpmt-- --compute --multievent 1

    2016-09-16 20:28:05.841 INFO  [3611] [TimesTable::dump@105] Opticks::postpropagate filter: OPropagator::
              3.402          0.410      30648.000        175.000 : OPropagator::prelaunch_0
              3.473          0.070      30648.000          0.000 : OPropagator::launch_0
    2016-09-16 20:28:05.842 INFO  [3611] [OpticksProfile::dump@143]  npy 20,1,4 /tmp/blyth/opticks/evt/PmtInBox/torch/Opticks.npy
    Opticks::postpropagate
       OptiXVersion :            3080

Huh trivial multievent fails but dumpseed multievent works::

   tpmt-- --compute --multievent 2 --dumpseed   ## works
   tpmt-- --compute --multievent 2 --trivial    ## hard CUDA crash at 2nd evt launch 

*trivial* gone one step beyond *dumpseed* in that it attempts to read from the genstep buffer
using the genstep_id read from the seed buffer.

Huh the below are working, but OKTest as used by tpmt is failing for gt 1::

   OpSeederTest --compute --dumpseed --multievent 2    ## huh it worked 
   OpSeederTest --compute --trivial  --multievent 2    ## huh it worked 
   OpSeederTest --compute --trivial --multievent 10    
       ##
       ## WHY IS THIS SUCCEEDING TO READ FROM GENSTEP ???
       ## ACTUALLY LOOKING MORE CLOSELY : THE OLD STUCK AT ZERO SEED ISSUE
       ## IS APPARENT FROM THE 2nd EVT 
       ##





See :doc:`optix_cuda_interop_3080` 




G4 
~~~

::


    delta:torch blyth$ grep ^CG4::propagate  DeltaTime.ini
    CG4::propagate_0=8.828125
    CG4::propagate_1=8.51953125
    CG4::propagate_2=9.1015625
    CG4::propagate_3=10.044921875
    CG4::propagate_4=10.056640625
    CG4::propagate_5=10.2578125
    CG4::propagate_6=10.111328125
    CG4::propagate_7=10.099609375
    CG4::propagate_8=10.140625
    CG4::propagate_9=10.322265625
    delta:torch blyth$ 
    delta:torch blyth$ 



::

    2016-09-16 14:32:26.858 INFO  [646185] [OpticksProfile::dump@129] Opticks::postpropagate dir /tmp/blyth/opticks/evt/PmtInBox/torch name Opticks.npy num_stamp 95
    2016-09-16 14:32:26.858 INFO  [646185] [TimesTable::dump@90] Opticks::postpropagate
               Time      DeltaTime             VM        DeltaVM
              0.000      23386.785          0.000       2650.000 : Opticks::Opticks_0
              0.621          0.621         66.000         66.000 : OpticksRun::OpticksRun_0
              0.693          0.072         68.000          2.000 : CG4::CG4_0
              2.543          1.850      30472.000      30404.000 : OpticksRun::createEvent_0
              2.816          0.273      30592.000        120.000 : _CG4::propagate_0
             11.645          8.828      34408.000       3816.000 : CG4::propagate_0
             11.645          0.000      34408.000          0.000 : _OpticksEvent::indexPhotonsCPU_0
             12.072          0.428      34436.000         28.000 : OpticksEvent::indexPhotonsCPU_0
             12.078          0.006      34436.000          0.000 : _OpSeeder::seedPhotonsFromGenstepsViaOptiX_0
             12.088          0.010      34438.000          2.000 : OpSeeder::seedPhotonsFromGenstepsViaOptiX_0
             12.088          0.000      34438.000          0.000 : _OPropagator::prelaunch_0
             16.533          4.445      34518.000         80.000 : OPropagator::prelaunch_0
             16.533          0.000      34518.000          0.000 : _OPropagator::launch_0
             17.357          0.824      34518.000          0.000 : OPropagator::launch_0
             20.881          3.523      34778.000        260.000 : OpticksRun::createEvent_1
             21.152          0.271      34898.000        120.000 : _CG4::propagate_1
             29.672          8.520      38678.000       3780.000 : CG4::propagate_1
             29.672          0.000      38678.000          0.000 : _OpticksEvent::indexPhotonsCPU_1
             30.104          0.432      38710.000         32.000 : OpticksEvent::indexPhotonsCPU_1
             30.105          0.002      38710.000          0.000 : _OpSeeder::seedPhotonsFromGenstepsViaOptiX_1
             30.107          0.002      38710.000          0.000 : OpSeeder::seedPhotonsFromGenstepsViaOptiX_1
             30.107          0.000      38710.000          0.000 : _OPropagator::launch_1
             30.902          0.795      38710.000          0.000 : OPropagator::launch_1
             34.254          3.352      38854.000        144.000 : OpticksRun::createEvent_2
             34.527          0.273      38974.000        120.000 : _CG4::propagate_2
             43.629          9.102      42808.000       3834.000 : CG4::propagate_2
             43.629          0.000      42808.000          0.000 : _OpticksEvent::indexPhotonsCPU_2
             44.051          0.422      42820.000         12.000 : OpticksEvent::indexPhotonsCPU_2
             44.053          0.002      42820.000          0.000 : _OpSeeder::seedPhotonsFromGenstepsViaOptiX_2
             44.055          0.002      42820.000          0.000 : OpSeeder::seedPhotonsFromGenstepsViaOptiX_2
             44.055          0.000      42820.000          0.000 : _OPropagator::launch_2
             44.840          0.785      42820.000          0.000 : OPropagator::launch_2
             48.547          3.707      42964.000        144.000 : OpticksRun::createEvent_3
             48.822          0.275      43084.000        120.000 : _CG4::propagate_3
             58.867         10.045      46856.000       3772.000 : CG4::propagate_3
             58.867          0.000      46856.000          0.000 : _OpticksEvent::indexPhotonsCPU_3
             59.311          0.443      46888.000         32.000 : OpticksEvent::indexPhotonsCPU_3
             59.312          0.002      46888.000          0.000 : _OpSeeder::seedPhotonsFromGenstepsViaOptiX_3
             59.314          0.002      46888.000          0.000 : OpSeeder::seedPhotonsFromGenstepsViaOptiX_3
             59.314          0.000      46888.000          0.000 : _OPropagator::launch_3
             60.148          0.834      46888.000          0.000 : OPropagator::launch_3
             64.523          4.375      47032.000        144.000 : OpticksRun::createEvent_4
             64.797          0.273      47152.000        120.000 : _CG4::propagate_4
             74.854         10.057      50924.000       3772.000 : CG4::propagate_4
             74.854          0.000      50924.000          0.000 : _OpticksEvent::indexPhotonsCPU_4
             75.299          0.445      50956.000         32.000 : OpticksEvent::indexPhotonsCPU_4
             75.299          0.000      50956.000          0.000 : _OpSeeder::seedPhotonsFromGenstepsViaOptiX_4
             75.303          0.004      50956.000          0.000 : OpSeeder::seedPhotonsFromGenstepsViaOptiX_4
             75.303          0.000      50956.000          0.000 : _OPropagator::launch_4
             76.096          0.793      50956.000          0.000 : OPropagator::launch_4
             80.383          4.287      51100.000        144.000 : OpticksRun::createEvent_5
             80.658          0.275      51220.000        120.000 : _CG4::propagate_5
             90.916         10.258      55129.000       3909.000 : CG4::propagate_5
             90.918          0.002      55129.000          0.000 : _OpticksEvent::indexPhotonsCPU_5
             91.359          0.441      55161.000         32.000 : OpticksEvent::indexPhotonsCPU_5
             91.359          0.000      55161.000          0.000 : _OpSeeder::seedPhotonsFromGenstepsViaOptiX_5
             91.363          0.004      55161.000          0.000 : OpSeeder::seedPhotonsFromGenstepsViaOptiX_5
             91.363          0.000      55161.000          0.000 : _OPropagator::launch_5
             92.133          0.770      55161.000          0.000 : OPropagator::launch_5
             96.373          4.240      55305.000        144.000 : OpticksRun::createEvent_6
             96.648          0.275      55425.000        120.000 : _CG4::propagate_6
            106.760         10.111      59198.000       3773.000 : CG4::propagate_6
            106.760          0.000      59198.000          0.000 : _OpticksEvent::indexPhotonsCPU_6
            107.236          0.477      59230.000         32.000 : OpticksEvent::indexPhotonsCPU_6
            107.236          0.000      59230.000          0.000 : _OpSeeder::seedPhotonsFromGenstepsViaOptiX_6
            107.240          0.004      59230.000          0.000 : OpSeeder::seedPhotonsFromGenstepsViaOptiX_6
            107.240          0.000      59230.000          0.000 : _OPropagator::launch_6
            108.045          0.805      59230.000          0.000 : OPropagator::launch_6
            112.213          4.168      59374.000        144.000 : OpticksRun::createEvent_7
            112.486          0.273      59494.000        120.000 : _CG4::propagate_7
            122.586         10.100      63266.000       3772.000 : CG4::propagate_7
            122.586          0.000      63266.000          0.000 : _OpticksEvent::indexPhotonsCPU_7
            123.035          0.449      63298.000         32.000 : OpticksEvent::indexPhotonsCPU_7
            123.035          0.000      63298.000          0.000 : _OpSeeder::seedPhotonsFromGenstepsViaOptiX_7
            123.039          0.004      63298.000          0.000 : OpSeeder::seedPhotonsFromGenstepsViaOptiX_7
            123.039          0.000      63298.000          0.000 : _OPropagator::launch_7
            123.799          0.760      63298.000          0.000 : OPropagator::launch_7
            128.016          4.217      63442.000        144.000 : OpticksRun::createEvent_8
            128.299          0.283      63562.000        120.000 : _CG4::propagate_8
            138.439         10.141      67335.000       3773.000 : CG4::propagate_8
            138.439          0.000      67335.000          0.000 : _OpticksEvent::indexPhotonsCPU_8
            138.916          0.477      67367.000         32.000 : OpticksEvent::indexPhotonsCPU_8
            138.918          0.002      67367.000          0.000 : _OpSeeder::seedPhotonsFromGenstepsViaOptiX_8
            138.920          0.002      67367.000          0.000 : OpSeeder::seedPhotonsFromGenstepsViaOptiX_8
            138.920          0.000      67367.000          0.000 : _OPropagator::launch_8
            139.711          0.791      67367.000          0.000 : OPropagator::launch_8
            143.928          4.217      67511.000        144.000 : OpticksRun::createEvent_9
            144.207          0.279      67631.000        120.000 : _CG4::propagate_9
            154.529         10.322      71407.000       3776.000 : CG4::propagate_9
            154.529          0.000      71407.000          0.000 : _OpticksEvent::indexPhotonsCPU_9
            154.977          0.447      71439.000         32.000 : OpticksEvent::indexPhotonsCPU_9
            154.979          0.002      71439.000          0.000 : _OpSeeder::seedPhotonsFromGenstepsViaOptiX_9
            154.980          0.002      71439.000          0.000 : OpSeeder::seedPhotonsFromGenstepsViaOptiX_9
            154.980          0.000      71439.000          0.000 : _OPropagator::launch_9
            155.746          0.766      71439.000          0.000 : OPropagator::launch_9
    2016-09-16 14:32:26.859 INFO  [646185] [OpticksProfile::dump@134]  npy 95,1,4 /tmp/blyth/opticks/evt/PmtInBox/torch/Opticks.npy
    2016-09-16 14:32:27.041 INFO  [646185] [Opticks::cleanup@1002] Opticks::cleanup
    2016-09-16 14:32:27.041 INFO  [646185] [CG4::cleanup@283] CG4::cleanup opening geometry

