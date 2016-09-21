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


3080 multievent was failing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Huh trivial multievent fails but dumpseed multievent works::

   tpmt-- --compute --multievent 2 --dumpseed   ## works
   tpmt-- --compute --multievent 2 --trivial    ## hard CUDA crash at 2nd evt launch 

*trivial* gone one step beyond *dumpseed* in that it attempts to read from the genstep buffer
using the genstep_id read from the seed buffer.

See :doc:`optix_cuda_interop_3080` 


3080 multievent times
~~~~~~~~~~~~~~~~~~~~~~~~~

::

    tpmt-;tpmt--  --compute --multievent 10    
   
     ## 0.5M launch takes around 0.2-0.25s with total of 2.5s between launches 
     ## 
     ## most of the 2.5s not needed in production...
     ## 
     ##              launch:0.25s      << fully needed in production
     ##
     ##       indexSequence:0.19,      << not needed
     ##       downloadEvent:~0.33      <<---  only partially needed, stream compaction grab hits
     ##           saveEvent:0.66,      << not needed
     ##            anaEvent:~0.90,     << not needed
     ##          resetEvent:0.25       << not needed
     ##
     ##         0.25+0.19+0.66+0.90+0.25+0.33 = 2.58
     ##


              0.000          25.575          0.000      31943.000          0.000 : OpticksRun::createEvent.BEG_9
              0.000          25.575          0.000      31943.000          0.000 : OpticksRun::createEvent.END_9
              0.001          25.576          0.001      31943.000          0.000 : OKPropagator::propagate.BEG_9
              0.000          25.576          0.000      31943.000          0.000 : _OEvent::upload_9
              0.000          25.576          0.000      31943.000          0.000 : OEvent::upload_9
              0.000          25.576          0.000      31943.000          0.000 : _OpSeeder::seedPhotonsFromGenstepsViaOptiX_9
              0.003          25.579          0.003      31943.000          0.000 : OpSeeder::seedPhotonsFromGenstepsViaOptiX_9
              0.000          25.579          0.000      31943.000          0.000 : _OPropagator::launch_9
              0.234          25.813          0.234      31943.000          0.000 : OPropagator::launch_9
              0.000          25.813          0.000      31943.000          0.000 : _OpIndexer::indexSequence_9
              0.190          26.004          0.190      31965.000         22.000 : OpIndexer::indexSequence_9
              0.007          26.011          0.007      31965.000          0.000 : OKPropagator::propagate.MID_9
              0.000          26.011          0.000      31965.000          0.000 : _OEvent::download_9
              0.316          26.327          0.316      32085.000        120.000 : OEvent::download_9
              0.000          26.327          0.000      32085.000          0.000 : OKPropagator::propagate.END_9
              0.000          26.327          0.000      32085.000          0.000 : OpticksRun::saveEvent.BEG_9
              0.684          27.011          0.684      32085.000          0.000 : OpticksRun::saveEvent.END_9
              0.000          27.011          0.000      32085.000          0.000 : OpticksRun::anaEvent.BEG_9
              0.886          27.896          0.886      32085.000          0.000 : OpticksRun::anaEvent.END_9
              0.000          27.896          0.000      32085.000          0.000 : OpticksRun::resetEvent.BEG_9
              0.251          28.147          0.251      32085.000          0.000 : OpticksRun::resetEvent.END_9
    2016-09-21 12:21:42.745 INFO  [291190] [OpticksProfile::dump@145]  npy 214,1,4 /tmp/blyth/opticks/evt/PmtInBox/torch/Opticks.npy
    2016-09-21 12:21:42.745 INFO  [291190] [OpticksProfile::dump@140] Opticks::postpropagate dir /tmp/blyth/opticks/evt/PmtInBox/torch name Opticks.npy num_stamp 214
    2016-09-21 12:21:42.746 INFO  [291190] [TimesTable::dump@105] Opticks::postpropagate filter: OPropagator::launch
              2.965           2.965          0.070      30656.000          0.000 : OPropagator::launch_0
              2.428           5.393          0.118      30798.000          0.000 : OPropagator::launch_1
              2.429           7.821          0.118      30940.000          0.000 : OPropagator::launch_2
              2.572          10.394          0.198      31091.000          0.000 : OPropagator::launch_3
              2.535          12.929          0.186      31233.000          0.000 : OPropagator::launch_4
              2.590          15.519          0.228      31375.000          0.000 : OPropagator::launch_5
              2.539          18.058          0.212      31517.000          0.000 : OPropagator::launch_6
              2.589          20.646          0.253      31659.000          0.000 : OPropagator::launch_7
              2.549          23.195          0.226      31801.000          0.000 : OPropagator::launch_8
              2.618          25.813          0.234      31943.000          0.000 : OPropagator::launch_9
    2016-09-21 12:21:42.746 INFO  [291190] [OpticksProfile::dump@145]  npy 214,1,4 /tmp/blyth/opticks/evt/PmtInBox/torch/Opticks.npy
    Opticks::postpropagate
       OptiXVersion :            3080
    /Users/blyth/opticks/bin/op.sh RC 0



Next Steps
------------

* *production* option to skip most processing expenses
* compaction in multievent setting 




Get compaction operational in multievent setting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Proof of concept code for stream compaction 

That uses a float4 buffer, and pulls back at float4 level.

For real usage photon buffer has shape (N,4,4) ie N* (4,4) items
need to pull back only the (4,4) photons matching some criteria. 


Demo code for stream compaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* env-;optixthrust-  OptiXThrust::compaction 

* https://bitbucket.org/simoncblyth/env/commits/398811a731ffc4caef3f07fdc18362b842d98c37 

Hmm this would be simpler with a float4x4 type ? 

env/optix/optixthrust/optixthrust_.cu::

    206 void OptiXThrust::compaction4()
    207 {
    208     // Initial approach of count_if/copy_if targeting 
    209     // a zip iterator output failed at runtime.
    210     //
    211     // But actually the approach adopted of using a mask 
    212     // is preferable as the mask has an obvious meaning 
    213     // and could be created by other means, such as with OptiX.
    214 
    215     printf("OptiXThrust::compaction4 size %u \n", m_size );
    216 
    217     assert(m_size % 4 == 0);
    218 
    219     // masker functor is passed the four float4 of each photon
    220     // demo code : just masking based on uint index encoded in 4th float4
    221     photon_mask4 mskr(5, 15);
    222 
    223     thrust::device_ptr<float4> d = getThrustDevicePtr<float4>(m_buffer, m_device);
    224 
    225     thrust::device_vector<int> mask(m_size/4) ;
    226 
    227     typedef thrust::device_vector<float4>::iterator Iterator;
    228     strided_range<Iterator> four0(d + 0, d + m_size, 4);
    229     strided_range<Iterator> four1(d + 1, d + m_size, 4);
    230     strided_range<Iterator> four2(d + 2, d + m_size, 4);
    231     strided_range<Iterator> four3(d + 3, d + m_size, 4);
    232 
    233     thrust::transform(   // step thru float4 buffer in groups of 4*float4 
    234           thrust::make_zip_iterator(thrust::make_tuple( four0.begin(), four1.begin(), four2.begin(), four3.begin() )),
    235           thrust::make_zip_iterator(thrust::make_tuple( four0.end(),   four1.end()  , four2.end()  , four3.end()   )),
    236           mask.begin(),
    237           mskr );
    238 
    239     //printf("mask\n"); thrust::copy(mask.begin(), mask.end(), std::ostream_iterator<int>(std::cout, "\n")); 
    240     unsigned int num = thrust::count(mask.begin(), mask.end(), 1);  // number of 1s in the mask
    241     if( num > 0 )
    242     {
    243         int* d_mask = thrust::raw_pointer_cast(mask.data());
    244         float4* photons = make_masked_buffer( d_mask, mask.size(), num );
    245         dump_photons( photons, num );
    246     }
    247 }











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

