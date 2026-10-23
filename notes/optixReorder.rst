notes/optixReorder
=====================


Overview
---------

Using optixReorder reduces optical photon simulation time by about a third -
but changing coherenceHint not seeming to make any difference.


Docs
-----

* https://forums.developer.nvidia.com/t/optix-8-0-release/263140
* https://www.nvidia.com/en-us/on-demand/session/other2023-optix8/

This small change from the optixTrace to the new optixTraverse, optixReorder,
optixInvoke device functions for SER, where the shader ID is given as coherency
hint bits to optixReorder, results in a 55% performance improvement (88 fps to
137 fps) of the scene_mdl_vMaterials.txt scene.  (Benchmarked under Windows 10
and 535.98 drivers on RTX 6000 Ada with interop 0 inside the
system_mdl_vMaterials.txt and the application’s benchmark mode (command line
option -m 1) measuring the raw raytracing performance without display.)


* https://github.com/NVIDIA/OptiX_Apps/blob/master/apps/MDL_renderer/shaders/raygeneration.cu#L177


To try
---------

1. DONE : production build - reduce state to minimize reorder overhead
2. DONE : increase OPTICKS_MAX_BOUNCE - no big effect
3. DONE : decrease OPTICKS_MAX_SLOT for the M100 - little diff
4. DONE : speedup render : added FPS and moved reorder into trace - used by all : simulate/simtrace/render
5. try other hints:

   * global CD radius
   * spatial position hash
   * spherical position hash
   * optixHitObject... accessors to inform the hint following traverse

6. DONE : optixTraverse + optixReorder + optixInvoke
7. DONE : optixir
8. try S4MultiLayrStack.h with simple scomplex.h scomplex_select.h - to avoid heavy thrust/complex.h compiled into kernel
9. try termination hint reordering


Overview
----------


::

    In [2]: 35/13
    Out[2]: 2.6923076923076925

Combined benefits of the below approaching a factor of 3 reduction in simulation kernel time

1. Release build
2. SER reordering
3. switch to optixir


Release : optixTraverse + optixReorder + optixInvoke - with .optixir
---------------------------------------------------------------------

* again small improvement for reordered when using .optixir vs .ptx


::

    sreport.desc_ranges.descTable  ( ta,tb : range begin,end timestamps expressed as seconds from first timestamp, ab: (tb-ta) )
    [NP::descTable_ {ranges} no-DT (33, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1792676405.967342 1792676405.974895              7553                 0                 2          0.007553                              init 
                             load_geom 1792676405.974895 1792676407.456333           1481438                 2                 3          1.481438                         load_geom 
                           upload_geom 1792676407.456367 1792676410.798313           3341946                 4                 5          3.341946                       upload_geom 
                         slice_genstep 1792676410.798607 1792676410.798629                22                 9                10          0.000022                     slice_genstep 
                  upload genstep slice 1792676410.798635 1792676410.820961             22326                11                12          0.022326              upload genstep slice 
                        simulate slice 1792676410.820961 1792676424.017905          13196944                12                13         13.196944                    simulate slice 
                        download slice 1792676424.017905 1792676424.382037            364132                13                14          0.364132                    download slice 
                        concat slices  1792676424.382105 1792676424.382123                18                15                16          0.000018                    concat slices  
                          save arrays  1792676424.382142 1792676433.748984           9366842                17                21          9.366842                      save arrays  
                         slice_genstep 1792676433.749101 1792676433.749114                13                23                24          0.000013                     slice_genstep 
                  upload genstep slice 1792676433.749120 1792676433.751703              2583                25                26          0.002583              upload genstep slice 
                        simulate slice 1792676433.751703 1792676448.209271          14457568                26                27         14.457568                    simulate slice 
                        download slice 1792676448.209271 1792676448.571729            362458                27                28          0.362458                    download slice 
                        concat slices  1792676448.571792 1792676448.571808                16                29                30          0.000016                    concat slices  
                          save arrays  1792676448.571819 1792676458.394380           9822561                31                35          9.822561                      save arrays  
                         slice_genstep 1792676458.394479 1792676458.394491                12                37                38          0.000012                     slice_genstep 
                  upload genstep slice 1792676458.394500 1792676458.397380              2880                39                40          0.002880              upload genstep slice 
                        simulate slice 1792676458.397380 1792676471.874467          13477087                40                41         13.477087                    simulate slice 
                        download slice 1792676471.874467 1792676472.237468            363001                41                42          0.363001                    download slice 
                        concat slices  1792676472.237520 1792676472.237537                17                43                44          0.000017                    concat slices  
                          save arrays  1792676472.237548 1792676481.823527           9585979                45                49          9.585979                      save arrays  
                         slice_genstep 1792676481.823660 1792676481.823673                13                51                52          0.000013                     slice_genstep 
                  upload genstep slice 1792676481.823679 1792676481.826265              2586                53                54          0.002586              upload genstep slice 
                        simulate slice 1792676481.826265 1792676495.410221          13583956                54                55         13.583956                    simulate slice 
                        download slice 1792676495.410221 1792676495.772388            362167                55                56          0.362167                    download slice 
                        concat slices  1792676495.772442 1792676495.772458                16                57                58          0.000016                    concat slices  
                          save arrays  1792676495.772470 1792676505.929661          10157191                59                63         10.157191                      save arrays  
                         slice_genstep 1792676505.929789 1792676505.929801                12                65                66          0.000012                     slice_genstep 
                  upload genstep slice 1792676505.929807 1792676505.932618              2811                67                68          0.002811              upload genstep slice 
                        simulate slice 1792676505.932618 1792676526.786723          20854105                68                69         20.854105                    simulate slice 
                        download slice 1792676526.786723 1792676527.149931            363208                69                70          0.363208                    download slice 
                        concat slices  1792676527.149983 1792676527.150000                17                71                72          0.000017                    concat slices  
                          save arrays  1792676527.150013 1792676537.250593          10100580                73                78         10.100580                      save arrays  
    num_timestamp 66 auto-offset from tmin 1792676405967342 2026-10-22T21:40:05.967
                                TOTAL: 59158323194045329 59158323325327387         131282058              1231              1281

     0 :                ta : ta




Release : optixTraverse + optixReorder + optixInvoke : (with .ptx)
----------------------------------------------------------------------

* HUH : slow "save arrays" in Release now ?

::

    [NP::descTable_ {ranges} no-DT (33, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1792591753.848337 1792591753.856096              7759                 0                 2          0.007759                              init 
                             load_geom 1792591753.856096 1792591755.301830           1445734                 2                 3          1.445734                         load_geom 
                           upload_geom 1792591755.301862 1792591755.545488            243626                 4                 5          0.243626                       upload_geom 
                         slice_genstep 1792591755.545754 1792591755.545776                22                 9                10          0.000022                     slice_genstep 
                  upload genstep slice 1792591755.545783 1792591755.568148             22365                11                12          0.022365              upload genstep slice 
                        simulate slice 1792591755.568148 1792591770.157064          14588916                12                13         14.588916                    simulate slice 
                        download slice 1792591770.157064 1792591770.520665            363601                13                14          0.363601                    download slice 
                        concat slices  1792591770.520732 1792591770.520751                19                15                16          0.000019                    concat slices  
                          save arrays  1792591770.520768 1792591781.182270          10661502                17                21         10.661502                      save arrays  
                         slice_genstep 1792591781.182374 1792591781.182387                13                23                24          0.000013                     slice_genstep 
                  upload genstep slice 1792591781.182393 1792591781.184976              2583                25                26          0.002583              upload genstep slice 
                        simulate slice 1792591781.184976 1792591796.118336          14933360                26                27         14.933360                    simulate slice 
                        download slice 1792591796.118336 1792591796.479859            361523                27                28          0.361523                    download slice 
                        concat slices  1792591796.479909 1792591796.479925                16                29                30          0.000016                    concat slices  
                          save arrays  1792591796.479936 1792591806.103335           9623399                31                35          9.623399                      save arrays  
                         slice_genstep 1792591806.109530 1792591806.109550                20                37                38          0.000020                     slice_genstep 
                  upload genstep slice 1792591806.109556 1792591806.112362              2806                39                40          0.002806              upload genstep slice 
                        simulate slice 1792591806.112362 1792591820.893809          14781447                40                41         14.781447                    simulate slice 
                        download slice 1792591820.893809 1792591821.254662            360853                41                42          0.360853                    download slice 
                        concat slices  1792591821.254712 1792591821.254729                17                43                44          0.000017                    concat slices  
                          save arrays  1792591821.254740 1792591832.094679          10839939                45                49         10.839939                      save arrays  
                         slice_genstep 1792591832.094779 1792591832.094792                13                51                52          0.000013                     slice_genstep 
                  upload genstep slice 1792591832.094798 1792591832.097385              2587                53                54          0.002587              upload genstep slice 
                        simulate slice 1792591832.097385 1792591846.732924          14635539                54                55         14.635539                    simulate slice 
                        download slice 1792591846.732924 1792591847.094881            361957                55                56          0.361957                    download slice 
                        concat slices  1792591847.094940 1792591847.094956                16                57                58          0.000016                    concat slices  
                          save arrays  1792591847.094966 1792591856.846110           9751144                59                63          9.751144                      save arrays  
                         slice_genstep 1792591856.846214 1792591856.846226                12                65                66          0.000012                     slice_genstep 
                  upload genstep slice 1792591856.846232 1792591856.848819              2587                67                68          0.002587              upload genstep slice 
                        simulate slice 1792591856.848819 1792591877.710448          20861629                68                69         20.861629                    simulate slice 
                        download slice 1792591877.710448 1792591878.070591            360143                69                70          0.360143                    download slice 
                        concat slices  1792591878.070642 1792591878.070658                16                71                72          0.000016                    concat slices  
                          save arrays  1792591878.070670 1792591887.967468           9896798                73                78          9.896798                      save arrays  
    num_timestamp 66 auto-offset from tmin 1792591753848337 2026-10-21T22:09:13.848
                                TOTAL: 59155529696829994 59155529830941955         134111961              1231              1281

     0 :                ta : ta
     1 :                tb : tb











Debug : optixTraverse + optixReorder + optixInvoke - with .optixir - PLUS AVOID SOME DUPLICATED payload pointer packing
-------------------------------------------------------------------------------------------------------------------------

* comparing with the below .ptx (only diff is .optixir cf .ptx) - non reorder same at 34, reorder evt shows slight improvment 23.5 -> ~22
* avoiding duplicated payload pointer - negligible effect


::

    [NP::descTable_ {ranges} no-DT (33, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1792736294.281676 1792736294.292131             10455                 0                 2          0.010455                              init 
                             load_geom 1792736294.292131 1792736296.182901           1890770                 2                 3          1.890770                         load_geom 
                           upload_geom 1792736296.182941 1792736301.529866           5346925                 4                 5          5.346925                       upload_geom 
                         slice_genstep 1792736301.530222 1792736301.530261                39                 9                10          0.000039                     slice_genstep 
                  upload genstep slice 1792736301.530268 1792736301.545767             15499                11                12          0.015499              upload genstep slice 
                        simulate slice 1792736301.545767 1792736323.015410          21469643                12                13         21.469643                    simulate slice 
                        download slice 1792736323.015410 1792736323.616422            601012                13                14          0.601012                    download slice 
                        concat slices  1792736323.616477 1792736323.616501                24                15                16          0.000024                    concat slices  
                          save arrays  1792736323.616526 1792736333.378881           9762355                17                21          9.762355                      save arrays  
                         slice_genstep 1792736333.379017 1792736333.379046                29                23                24          0.000029                     slice_genstep 
                  upload genstep slice 1792736333.379053 1792736333.381747              2694                25                26          0.002694              upload genstep slice 
                        simulate slice 1792736333.381747 1792736356.234465          22852718                26                27         22.852718                    simulate slice 
                        download slice 1792736356.234465 1792736356.837892            603427                27                28          0.603427                    download slice 
                        concat slices  1792736356.837948 1792736356.837969                21                29                30          0.000021                    concat slices  
                          save arrays  1792736356.837982 1792736366.572690           9734708                31                35          9.734708                      save arrays  
                         slice_genstep 1792736366.572835 1792736366.572862                27                37                38          0.000027                     slice_genstep 
                  upload genstep slice 1792736366.572870 1792736366.575520              2650                39                40          0.002650              upload genstep slice 
                        simulate slice 1792736366.575520 1792736388.435986          21860466                40                41         21.860466                    simulate slice 
                        download slice 1792736388.435986 1792736389.038814            602828                41                42          0.602828                    download slice 
                        concat slices  1792736389.038864 1792736389.038885                21                43                44          0.000021                    concat slices  
                          save arrays  1792736389.038901 1792736399.013409           9974508                45                49          9.974508                      save arrays  
                         slice_genstep 1792736399.013574 1792736399.013606                32                51                52          0.000032                     slice_genstep 
                  upload genstep slice 1792736399.013613 1792736399.016220              2607                53                54          0.002607              upload genstep slice 
                        simulate slice 1792736399.016220 1792736420.956959          21940739                54                55         21.940739                    simulate slice 
                        download slice 1792736420.956959 1792736421.556375            599416                55                56          0.599416                    download slice 
                        concat slices  1792736421.556421 1792736421.556442                21                57                58          0.000021                    concat slices  
                          save arrays  1792736421.556455 1792736431.316727           9760272                59                63          9.760272                      save arrays  
                         slice_genstep 1792736431.316929 1792736431.316958                29                65                66          0.000029                     slice_genstep 
                  upload genstep slice 1792736431.316965 1792736431.319577              2612                67                68          0.002612              upload genstep slice 
                        simulate slice 1792736431.319577 1792736465.422925          34103348                68                69         34.103348                    simulate slice 
                        download slice 1792736465.422925 1792736466.021456            598531                69                70          0.598531                    download slice 
                        concat slices  1792736466.021512 1792736466.021533                21                71                72          0.000021                    concat slices  
                          save arrays  1792736466.021546 1792736476.374325          10352779                73                78         10.352779                      save arrays  
    num_timestamp 66 auto-offset from tmin 1792736294281676 2026-10-23T14:18:14.281
                                TOTAL: 59160300248429302 59160300430520528         182091226              1231              1281

     0 :                ta : ta
















Debug : optixTraverse + optixReorder + optixInvoke : with .ptx
-----------------------------------------------------------------

::


    [NP::descTable_ {ranges} no-DT (33, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1792591154.464678 1792591154.474867             10189                 0                 2          0.010189                              init 
                             load_geom 1792591154.474867 1792591156.267942           1793075                 2                 3          1.793075                         load_geom 
                           upload_geom 1792591156.267978 1792591161.538500           5270522                 4                 5          5.270522                       upload_geom 
                         slice_genstep 1792591161.538847 1792591161.538887                40                 9                10          0.000040                     slice_genstep 
                  upload genstep slice 1792591161.538893 1792591161.554292             15399                11                12          0.015399              upload genstep slice 
                        simulate slice 1792591161.554292 1792591184.736237          23181945                12                13         23.181945                    simulate slice 
                        download slice 1792591184.736237 1792591185.102755            366518                13                14          0.366518                    download slice 
                        concat slices  1792591185.102826 1792591185.102850                24                15                16          0.000024                    concat slices  
                          save arrays  1792591185.102874 1792591195.547336          10444462                17                21         10.444462                      save arrays  
                         slice_genstep 1792591195.547468 1792591195.547495                27                23                24          0.000027                     slice_genstep 
                  upload genstep slice 1792591195.547502 1792591195.550102              2600                25                26          0.002600              upload genstep slice 
                        simulate slice 1792591195.550102 1792591219.041664          23491562                26                27         23.491562                    simulate slice 
                        download slice 1792591219.041664 1792591219.410623            368959                27                28          0.368959                    download slice 
                        concat slices  1792591219.410701 1792591219.410723                22                29                30          0.000022                    concat slices  
                          save arrays  1792591219.410736 1792591229.397633           9986897                31                35          9.986897                      save arrays  
                         slice_genstep 1792591229.397783 1792591229.397812                29                37                38          0.000029                     slice_genstep 
                  upload genstep slice 1792591229.397818 1792591229.400565              2747                39                40          0.002747              upload genstep slice 
                        simulate slice 1792591229.400565 1792591252.989421          23588856                40                41         23.588856                    simulate slice 
                        download slice 1792591252.989421 1792591253.347332            357911                41                42          0.357911                    download slice 
                        concat slices  1792591253.347392 1792591253.347413                21                43                44          0.000021                    concat slices  
                          save arrays  1792591253.347426 1792591263.533943          10186517                45                49         10.186517                      save arrays  
                         slice_genstep 1792591263.534108 1792591263.534137                29                51                52          0.000029                     slice_genstep 
                  upload genstep slice 1792591263.534144 1792591263.536749              2605                53                54          0.002605              upload genstep slice 
                        simulate slice 1792591263.536749 1792591287.212430          23675681                54                55         23.675681                    simulate slice 
                        download slice 1792591287.212430 1792591287.570916            358486                55                56          0.358486                    download slice 
                        concat slices  1792591287.570968 1792591287.570989                21                57                58          0.000021                    concat slices  
                          save arrays  1792591287.571002 1792591297.395819           9824817                59                63          9.824817                      save arrays  
                         slice_genstep 1792591297.395998 1792591297.396025                27                65                66          0.000027                     slice_genstep 
                  upload genstep slice 1792591297.396032 1792591297.398642              2610                67                68          0.002610              upload genstep slice 
                        simulate slice 1792591297.398642 1792591331.413899          34015257                68                69         34.015257                    simulate slice 
                        download slice 1792591331.413899 1792591331.774892            360993                69                70          0.360993                    download slice 
                        concat slices  1792591331.774946 1792591331.774965                19                71                72          0.000019                    concat slices  
                          save arrays  1792591331.774978 1792591341.672853           9897875                73                78          9.897875                      save arrays  
    num_timestamp 66 auto-offset from tmin 1792591154464678 2026-10-21T21:59:14.464
                                TOTAL: 59155510737283966 59155510924490708         187206742              1231              1281

     0 :                ta : ta
     1 :                tb : tb








cxs_min.sh "ser_medium_scan" : all events are M100 torch photons from CD center - using Ada 5000 on workstation
----------------------------------------------------------------------------------------------------------------

::

    elif [ "$TEST" == "ser_medium_scan" ]; then 


       ser_path=/tmp/$USER/opticks/$name/SER.txt
       mkdir -p $(dirname $ser_path)
       export OPTICKS_SER=filepath:$ser_path
       cat << EOS > $ser_path
    hintMethod:0,coherenceHintBits:4,bounceInterval:1   # optixReorder()
    hintMethod:1,coherenceHintBits:4,bounceInterval:4   # optixReorder( bounce / params.SER_bounceInterval, params.SER_coherenceHintBits )
    hintMethod:2,coherenceHintBits:4,bounceInterval:1   # optixReorder( ctx.p.flagidx()                   , params.SER_coherenceHintBits )
    hintMethod:3,coherenceHintBits:4,bounceInterval:1   # optixReorder(0,0)
    hintMethod:-1,coherenceHintBits:0,bounceInterval:0  # bounceInterval:0 disables the reorder
    EOS

       opticks_num_event=$(wc -l < $ser_path)
       opticks_num_photon=M100x$opticks_num_event
       opticks_num_genstep=1x$opticks_num_event
       opticks_running_mode=SRM_TORCH


When reorder is enabled get useful reduction in simulation time (from 32s to 22s) - but seems the coherence hints are doing nothing::

    TEST=ser_medium_scan cxs_min.sh
    ...

    .sreport.desc_runprof.descTable 
    [NP::descTable_ {runprof} no-DT (10, 3, )
                                                  st[us]            vm[kb]            rs[kb]          TS_DELTA
                   A000_SEvt__setIndex 1792482552.667707           8114384           1251460                 -               A000_SEvt__setIndex 
                   A000_SEvt__endIndex 1792482574.815569          27798732           2592024         22.147862               A000_SEvt__endIndex 
                   A001_SEvt__setIndex 1792482575.435044          26464464           1258076          0.619475               A001_SEvt__setIndex 
                   A001_SEvt__endIndex 1792482598.002554          27798280           2592196         22.567510               A001_SEvt__endIndex 
                   A002_SEvt__setIndex 1792482598.603818          26464464           1258508          0.601264               A002_SEvt__setIndex 
                   A002_SEvt__endIndex 1792482620.664792          27798652           2592312         22.060974               A002_SEvt__endIndex 
                   A003_SEvt__setIndex 1792482621.271955          26464464           1258148          0.607163               A003_SEvt__setIndex 
                   A003_SEvt__endIndex 1792482644.211676          27798536           2592220         22.939721               A003_SEvt__endIndex 
                   A004_SEvt__setIndex 1792482644.794806          26464464           1258148          0.583130               A004_SEvt__setIndex 
                   A004_SEvt__endIndex 1792482677.400512          27798576           2592260         32.605706               A004_SEvt__endIndex 
    num_timestamp 10 auto-offset from tmin 1792482552667707 2026-10-20T15:49:12.667
                                TOTAL: 17924826107868433         252965016          19245352

     0 :            st[us] : st[us]



With max_slot M100 - little diff::

    [NP::descTable_ {runprof} no-DT (10, 3, )
                                                  st[us]            vm[kb]            rs[kb]          TS_DELTA
                   A000_SEvt__setIndex 1792483522.121056           8114384           1255084                 -               A000_SEvt__setIndex 
                   A000_SEvt__endIndex 1792483544.063875          16788684           2595112         21.942819               A000_SEvt__endIndex 
                   A001_SEvt__setIndex 1792483553.850582          15454416           1260868          9.786707               A001_SEvt__setIndex 
                   A001_SEvt__endIndex 1792483576.149978          16788232           2594684         22.299396               A001_SEvt__endIndex 
                   A002_SEvt__setIndex 1792483585.433412          15454416           1260960          9.283434               A002_SEvt__setIndex 
                   A002_SEvt__endIndex 1792483607.113480          16788604           2595580         21.680068               A002_SEvt__endIndex 
                   A003_SEvt__setIndex 1792483616.618149          15454416           1261392          9.504669               A003_SEvt__setIndex 
                   A003_SEvt__endIndex 1792483639.155299          16788488           2595760         22.537150               A003_SEvt__endIndex 
                   A004_SEvt__setIndex 1792483648.833681          15454416           1261688          9.678382               A004_SEvt__setIndex 
                   A004_SEvt__endIndex 1792483680.876128          16788528           2595800         32.042447               A004_SEvt__endIndex 
    num_timestamp 10 auto-offset from tmin 1792483522121056 2026-10-20T16:05:22.121
                                TOTAL: 17924835974215640         153874584          19276928


With max_bounce 63 - no big effect, 22/34=0.65::

    .sreport.desc_runprof.descTable 
    [NP::descTable_ {runprof} no-DT (10, 3, )
                                                  st[us]            vm[kb]            rs[kb]          TS_DELTA
                   A000_SEvt__setIndex 1792483860.665529           8114384           1256060                 -               A000_SEvt__setIndex 
                   A000_SEvt__endIndex 1792483883.230971          16815564           2623680         22.565442               A000_SEvt__endIndex 
                   A001_SEvt__setIndex 1792483893.776332          15454416           1262532         10.545361               A001_SEvt__setIndex 
                   A001_SEvt__endIndex 1792483916.666100          16815196           2623336         22.889768               A001_SEvt__endIndex 
                   A002_SEvt__setIndex 1792483926.636529          15454416           1262556          9.970429               A002_SEvt__setIndex 
                   A002_SEvt__endIndex 1792483948.869227          16815440           2623580         22.232698               A002_SEvt__endIndex 
                   A003_SEvt__setIndex 1792483958.439824          15454416           1262556          9.570597               A003_SEvt__setIndex 
                   A003_SEvt__endIndex 1792483981.539297          16815312           2623452         23.099473               A003_SEvt__endIndex 
                   A004_SEvt__setIndex 1792483991.124314          15454416           1262556          9.585017               A004_SEvt__setIndex 
                   A004_SEvt__endIndex 1792484025.002064          16815404           2623544         33.877750               A004_SEvt__endIndex 
    num_timestamp 10 auto-offset from tmin 1792483860665529 2026-10-20T16:11:00.665
                                TOTAL: 17924839385950187         154008964          19423852

     0 :            st[us] : st[us]
     1 :            vm[kb] : vm[kb]
     2 :            rs[kb] : rs[kb]
     tmin 1792483860665529 2026-10-20T16:11:00.665
     tmax 1792484025002064 2026-10-20T16:13:45.002
     delt        164336535       164.336535



Production build, significant impact  15/21=0.71::

    report.desc_runprof.descTable 
    [NP::descTable_ {runprof} no-DT (10, 3, )
                                                  st[us]            vm[kb]            rs[kb]          TS_DELTA
                   A000_SEvt__setIndex 1792497640.691033           8141708           1229088                 -               A000_SEvt__setIndex 
                   A000_SEvt__endIndex 1792497655.576534          16842900           2593824         14.885501               A000_SEvt__endIndex 
                   A001_SEvt__setIndex 1792497656.154181          15481740           1232664          0.577647               A001_SEvt__setIndex 
                   A001_SEvt__endIndex 1792497671.388321          16842512           2593604         15.234140               A001_SEvt__endIndex 
                   A002_SEvt__setIndex 1792497672.058058          15481740           1232832          0.669737               A002_SEvt__setIndex 
                   A002_SEvt__endIndex 1792497686.623690          16842780           2594140         14.565632               A002_SEvt__endIndex 
                   A003_SEvt__setIndex 1792497687.272957          15481740           1233100          0.649267               A003_SEvt__setIndex 
                   A003_SEvt__endIndex 1792497702.611586          16842616           2594020         15.338629               A003_SEvt__endIndex 
                   A004_SEvt__setIndex 1792497703.280051          15481740           1233144          0.668465               A004_SEvt__setIndex 
                   A004_SEvt__endIndex 1792497724.629101          16842716           2593928         21.349050               A004_SEvt__endIndex 
    num_timestamp 10 auto-offset from tmin 1792497640691033 2026-10-20T20:00:40.691
                                TOTAL: 17924976800285512         154282192          19130344

     0 :            st[us] : st[us]



    [NP::descTable_ {ranges} no-DT (33, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1792497633.035551 1792497633.043238              7687                 0                 2          0.007687                              init 
                             load_geom 1792497633.043238 1792497634.504547           1461309                 2                 3          1.461309                         load_geom 
                           upload_geom 1792497634.504803 1792497640.690812           6186009                 4                 5          6.186009                       upload_geom 
                         slice_genstep 1792497640.691098 1792497640.691122                24                 9                10          0.000024                     slice_genstep 
                  upload genstep slice 1792497640.691128 1792497640.713466             22338                11                12          0.022338              upload genstep slice 
                        simulate slice 1792497640.713466 1792497655.211095          14497629                12                13         14.497629                    simulate slice 
                        download slice 1792497655.211095 1792497655.576413            365318                13                14          0.365318                    download slice 
                        concat slices  1792497655.576469 1792497655.576489                20                15                16          0.000020                    concat slices  
                          save arrays  1792497655.576508 1792497656.154064            577556                17                21          0.577556                      save arrays  
                         slice_genstep 1792497656.154225 1792497656.154238                13                23                24          0.000013                     slice_genstep 
                  upload genstep slice 1792497656.154244 1792497656.156970              2726                25                26          0.002726              upload genstep slice 
                        simulate slice 1792497656.156970 1792497671.025224          14868254                26                27         14.868254                    simulate slice 
                        download slice 1792497671.025224 1792497671.388209            362985                27                28          0.362985                    download slice 
                        concat slices  1792497671.388272 1792497671.388288                16                29                30          0.000016                    concat slices  
                          save arrays  1792497671.388299 1792497672.032604            644305                31                35          0.644305                      save arrays  
                         slice_genstep 1792497672.058125 1792497672.058142                17                37                38          0.000017                     slice_genstep 
                  upload genstep slice 1792497672.058148 1792497672.060735              2587                39                40          0.002587              upload genstep slice 
                        simulate slice 1792497672.060735 1792497686.262012          14201277                40                41         14.201277                    simulate slice 
                        download slice 1792497686.262012 1792497686.623579            361567                41                42          0.361567                    download slice 
                        concat slices  1792497686.623642 1792497686.623658                16                43                44          0.000016                    concat slices  
                          save arrays  1792497686.623669 1792497687.268948            645279                45                49          0.645279                      save arrays  
                         slice_genstep 1792497687.273005 1792497687.273019                14                51                52          0.000014                     slice_genstep 
                  upload genstep slice 1792497687.273025 1792497687.275608              2583                53                54          0.002583              upload genstep slice 
                        simulate slice 1792497687.275608 1792497702.184684          14909076                54                55         14.909076                    simulate slice 
                        download slice 1792497702.184684 1792497702.611485            426801                55                56          0.426801                    download slice 
                        concat slices  1792497702.611539 1792497702.611555                16                57                58          0.000016                    concat slices  
                          save arrays  1792497702.611566 1792497703.252313            640747                59                63          0.640747                      save arrays  
                         slice_genstep 1792497703.280124 1792497703.280139                15                65                66          0.000015                     slice_genstep 
                  upload genstep slice 1792497703.280145 1792497703.282719              2574                67                68          0.002574              upload genstep slice 
                        simulate slice 1792497703.282719 1792497724.207686          20924967                68                69         20.924967                    simulate slice 
                        download slice 1792497724.207686 1792497724.629004            421318                69                70          0.421318                    download slice 
                        concat slices  1792497724.629050 1792497724.629065                15                71                72          0.000015                    concat slices  
                          save arrays  1792497724.629076 1792497725.246814            617738                73                78          0.617738                      save arrays  
    num_timestamp 66 auto-offset from tmin 1792497633035551 2026-10-20T20:00:33.035
                                TOTAL: 59152423299535148 59152423391687944          92152796              1231              1281

     0 :                ta : ta
     1 :                tb : tb
     2 :                ab : ab


HUH compare back with Debug, notice that "save arrays" is factor of 10-20x slower for the Debug build ?

* ~/o/notes/issues/cxs_min_save_arrays_very_slow_in_Debug_build.rst - taking about 10s(0.6s) in Debug(Release)

::

    [NP::descTable_ {ranges} no-DT (33, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1792498487.262349 1792498487.272743             10394                 0                 2          0.010394                              init 
                             load_geom 1792498487.272743 1792498489.072099           1799356                 2                 3          1.799356                         load_geom 
                           upload_geom 1792498489.072135 1792498489.403941            331806                 4                 5          0.331806                       upload_geom 
                         slice_genstep 1792498489.404270 1792498489.404306                36                 9                10          0.000036                     slice_genstep 
                  upload genstep slice 1792498489.404312 1792498489.419749             15437                11                12          0.015437              upload genstep slice 
                        simulate slice 1792498489.419749 1792498511.858490          22438741                12                13         22.438741                    simulate slice 
                        download slice 1792498511.858490 1792498512.345051            486561                13                14          0.486561                    download slice 
                        concat slices  1792498512.345119 1792498512.345141                22                15                16          0.000022                    concat slices  
                          save arrays  1792498512.345165 1792498522.570141          10224976                17                21         10.224976                      save arrays  
                         slice_genstep 1792498522.570281 1792498522.570309                28                23                24          0.000028                     slice_genstep 
                  upload genstep slice 1792498522.570315 1792498522.572918              2603                25                26          0.002603              upload genstep slice 
                        simulate slice 1792498522.572918 1792498545.371080          22798162                26                27         22.798162                    simulate slice 
                        download slice 1792498545.371080 1792498545.845344            474264                27                28          0.474264                    download slice 
                        concat slices  1792498545.845403 1792498545.845423                20                29                30          0.000020                    concat slices  
                          save arrays  1792498545.845436 1792498555.597903           9752467                31                35          9.752467                      save arrays  
                         slice_genstep 1792498555.598044 1792498555.598072                28                37                38          0.000028                     slice_genstep 
                  upload genstep slice 1792498555.598087 1792498555.600683              2596                39                40          0.002596              upload genstep slice 
                        simulate slice 1792498555.600683 1792498577.763314          22162631                40                41         22.162631                    simulate slice 
                        download slice 1792498577.763314 1792498578.215727            452413                41                42          0.452413                    download slice 
                        concat slices  1792498578.215786 1792498578.215807                21                43                44          0.000021                    concat slices  
                          save arrays  1792498578.215820 1792498587.833406           9617586                45                49          9.617586                      save arrays  
                         slice_genstep 1792498587.838682 1792498587.838711                29                51                52          0.000029                     slice_genstep 
                  upload genstep slice 1792498587.838717 1792498587.841317              2600                53                54          0.002600              upload genstep slice 
                        simulate slice 1792498587.841317 1792498610.889796          23048479                54                55         23.048479                    simulate slice 
                        download slice 1792498610.889796 1792498611.332494            442698                55                56          0.442698                    download slice 
                        concat slices  1792498611.332547 1792498611.332568                21                57                58          0.000021                    concat slices  
                          save arrays  1792498611.332581 1792498620.965751           9633170                59                63          9.633170                      save arrays  
                         slice_genstep 1792498620.965914 1792498620.965943                29                65                66          0.000029                     slice_genstep 
                  upload genstep slice 1792498620.965950 1792498620.968551              2601                67                68          0.002601              upload genstep slice 
                        simulate slice 1792498620.968551 1792498655.068900          34100349                68                69         34.100349                    simulate slice 
                        download slice 1792498655.068900 1792498655.512901            444001                69                70          0.444001                    download slice 
                        concat slices  1792498655.512958 1792498655.512978                20                71                72          0.000020                    concat slices  
                          save arrays  1792498655.512991 1792498666.170623          10657632                73                78         10.657632                      save arrays  
    num_timestamp 66 auto-offset from tmin 1792498487262349 2026-10-20T20:14:47.262
                                TOTAL: 59152452500220403 59152452679122180         178901777              1231              1281





API from OptiX 8
-----------------

* https://raytracing-docs.nvidia.com/optix8/api/group__optix__device__api.html

::

    static __forceinline__ __device__ void 	optixReorder ()
    static __forceinline__ __device__ void 	optixReorder (unsigned int coherenceHint, unsigned int numCoherenceHintBitsFromLSB)


numCoherenceHintBitsFromLSB
   eg 4 : look at least significant 4 bits of coherenceHint 


ideas for coherenceHint
---------------------------

* p.flag index : added p.flagidx() [0->15] for this
* global position radius
* form some grouped boundary, or facilitate lookup imat/omat and use that ?

Alternates to try for the coherenceHint:

1. ctx.p.time based time_bins ?
2. 3-bit boundary group combined with 3-bit temporal bin ?





how to get the flag index : added sphoton::flagidx [0->15]
-----------------------------------------------------------



coherenceHint
----------------

* https://github.com/BNLNPPS/simphony/blob/feat/ser/CSGOptiX/CSGOptiX7.cu

::

    #if OPTIX_VERSION >= 80000
            if (params.SerInterval > 0 && (bounce % params.SerInterval) == 0)
            {
                unsigned ser_key = 0u;
                switch (params.SerHint)
                {
                case 1:
                    ser_key = bounce / params.SerInterval;
                    break;
                case 2:
                    ser_key = ((ctx.p.boundary() & 7u) << 1) | (ctx.p.wavelength < 200.f ? 1u : 0u);
                    break;
                }
                optixReorder(ser_key, 4);
            }
    #endif



The ctx.p.boundary is the boundary index that establishes unique combinations
of inner and outer material and surface indices - the ordering is not
controlled and there are typically a few hundred distinct boundary indices in a
geometry. So trying to use three bits from it would seem to be rather random
and unlikely to yield any useful coherence ? Do you have counter arguments ?
Any suggestions for what else could be used to better provide a coherence hint
?

The primary cause of divergence is that the photon lifetimes can be very
different. Photons right next to each other in the same volume and about to hit
the same surface can have very different lifetimes. Doesnt that suggest that
spatial grouping would be less useful than temporal grouping ?

You have hit on the core tension between traditional computer graphics ray
tracing and Monte Carlo particle transport. Your observation is entirely
correct: in optical simulations, spatial proximity does not guarantee
similarity in history, state, or remaining lifespan.

This exact reality explains why temporal and lifecycle grouping (Hint #1) is
far more important for Monte Carlo simulation than pure spatial grouping.

The Two Competing Forces: Trace vs. Transport

To understand why your insight is spot-on, it helps to look at what happens
during a single iteration of the while loop:


1. The trace() Phase (Ray-Geometry Intersection):

   * This is handled directly by the OptiX hardware ray tracer.

   * What it cares about: Spatial locality. If two photons are physically close to
     each other—even if one is on bounce 2 and the other is on bounce 50—their rays
     traverse the same bounding volume hierarchy (BVH) nodes. Grouping them
     spatially ensures that the GPU cores fetch the same acceleration structure
     nodes from memory, maximizing cache hits.

2. The propagate() & Branching Phase (Physics / State Updates):

   * This is handled by custom CUDA code (propagate logic, scattering,
     absorption checks, boundary handling).

   * What it cares about: Lifecycle and branch coherence. If a warp contains a mix
     of newly born photons and photons about to terminate, the warp suffers from
     massive thread divergence and idle lanes. Furthermore, if photon lifetimes
     diverge wildly, dead threads begin to accumulate in warps, starving the GPU of
     active work.





Cosmic muon simulations with many hundreds of millions or even billions of
photons is a important use case of Opticks - is trying 6 bits worthwhile ?

At the scale of hundreds of millions or billions of concurrent photons (such as
cosmic muon bundles passing through a massive detector), the calculus changes
completely.

In that regime, trying 6 bits (64 bins) is absolutely worthwhile.


The primary risk of a higher bin count (like 6 bits) is that bins become
under-populated, leaving fewer than 32 threads per bin and causing warp
starvation. However, when your initial launch size is 1 billion threads

::

    // 6-bit breakdown: 3 bits temporal, 3 bits material/state category
    unsigned int time_bin = (bounce / params.SerInterval) & 0x7;          // 3 bits (8 temporal steps)
    unsigned int state_bin = get_high_level_physics_state(ctx.p) & 0x7;    // 3 bits (8 physics states)
    unsigned int ser_key  = state_bin | (time_bin << 3);                 // Combined 6-bit key



The below is an extract from the CUDA kernel of the Simphony Opticks fork.
Please explain the optixReorder usage::

    static __forceinline__ __device__ void simulate( const uint3& launch_idx, const uint3& dim, quad2* prd )
    {
        sevent* evt = params.evt ;
        if (launch_idx.x >= evt->num_seed) return;   // was evt->num_photon

        unsigned idx = launch_idx.x ;
        unsigned genstep_idx = evt->seed[idx] ;
        const quad6& gs = evt->genstep[genstep_idx] ;
        // genstep needs the raw index, from zero for each genstep slice sub-launch

        unsigned long long photon_idx = params.photon_slot_offset + idx ;
        // 2025/10/20 change from unsigned to avoid clocking photon_idx and duplicating
        //
        // rng_state access and array recording needs the absolute photon_idx
        // for multi-launch and single-launch simulation to match.
        // The offset hides the technicality of the multi-launch from output.

        qsim* sim = params.sim ;

        RNG rng ;
        sim->rng->init( rng, sim->evt->index, photon_idx );

        sctx ctx = {} ;
        ctx.evt = evt ;   // sevent.h
        ctx.prd = prd ;   // squad.h quad2

        ctx.idx = idx ;
        ctx.pidx = photon_idx ;

    #if !defined(PRODUCTION) && defined(DEBUG_PIDX)
        ctx.pidx_debug = sim->base->pidx == photon_idx ;
    #endif

        sim->generate_photon(ctx.p, rng, gs, photon_idx, genstep_idx );
        // Only Cerenkov and scintillation gensteps store a source-material line in
        // q0.u.z. TORCH and FRAME use that slot for a genstep id, so leave the
        // carried line invalid for all other genstep types.
        const unsigned gentype = gs.q0.u.x;
        const bool carries_matline = CSGOptiX7_GenstepCarriesMaterialLine(gentype);
        ctx.current_matline = carries_matline ? gs.q0.u.z : 0xFFFFFFFFu;

        FlowAction command = FlowAction::Start;
        int bounce = 0 ;
        if(evt->record) ctx.point(bounce);
        while( bounce < evt->max_bounce && ctx.p.time < params.max_time )
        {
            float tmin = ( ctx.p.orient_boundary_flag & params.PropagateEpsilon0Mask ) ? params.tmin0 : params.tmin ;

    #if OPTIX_VERSION >= 80000
            if (params.SerInterval > 0 && (bounce % params.SerInterval) == 0)
            {
                unsigned ser_key = 0u;
                switch (params.SerHint)
                {
                case 1:
                    ser_key = bounce / params.SerInterval;
                    break;
                case 2:
                    ser_key = ((ctx.p.boundary() & 7u) << 1) | (ctx.p.wavelength < 200.f ? 1u : 0u);
                    break;
                }
                optixReorder(ser_key, 4);
            }
    #endif

            // intersect query filling (quad2)prd
            switch(params.PropagateRefine)
            {
                case 0u: trace<false>( params.handle, ctx.p.pos, ctx.p.mom, tmin, params.tmax, prd, params.vizmask, params.PropagateRefineDistance );  break ;
                case 1u: trace<true>(  params.handle, ctx.p.pos, ctx.p.mom, tmin, params.tmax, prd, params.vizmask, params.PropagateRefineDistance );  break ;
            }

            if( prd->boundary() == 0xffffu ) break ; // SHOULD ONLY HAPPEN FOR PHOTONS STARTING OUTSIDE WORLD
            // propagate can do nothing meaningful without a boundary

            // HMM: normalize here or within CSG ? Actually only needed for
            // geometry with active scaling, such as ellipsoid.
            // TODO: move this so its only done when needed
            //     ~/o/notes/issues/CSGOptiX_simulate_avoid_normalizing_every_normal.rst
            //

            float3* normal = prd->normal();
            *normal = normalize(*normal);

    #ifndef PRODUCTION
            ctx.trace(bounce);
    #endif
            command = sim->propagate(bounce, rng, ctx);
            bounce++;
            if(evt->record) ctx.point(bounce) ;
            if(command == FlowAction::Break) break;
        }
    #ifndef PRODUCTION
        ctx.end();  // write seq, tag, flat
    #endif


        if( evt->photon )
        {
            evt->photon[idx] = ctx.p ;  // *idx* (not *photon_idx*) as needs to go from zero for photons from a slice of genstep array
        }









Initial Check (lo) : medium_evt : 100M photons 10 genstep
----------------------------------------------------------

::

    TEST=medium_evt cxs_min.sh

    ...
    sreport.desc_ranges ranges : (9, 5, )
    .sreport.desc_ranges.descTable  ( ta,tb : range begin,end timestamps expressed as seconds from first timestamp, ab: (tb-ta) )
    [NP::descTable_ {ranges} no-DT (9, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1792395314.727872 1792395314.737066              9194                 0                 2          0.009194                              init 
                             load_geom 1792395314.737066 1792395316.543765           1806699                 2                 3          1.806699                         load_geom 
                           upload_geom 1792395316.543805 1792395316.896839            353034                 4                 5          0.353034                       upload_geom 
                         slice_genstep 1792395316.897158 1792395316.897194                36                 9                10          0.000036                     slice_genstep 
                  upload genstep slice 1792395316.897201 1792395316.932829             35628                11                12          0.035628              upload genstep slice 
                        simulate slice 1792395316.932829 1792395348.042201          31109372                12                13         31.109372                    simulate slice 
                        download slice 1792395348.042201 1792395348.401072            358871                13                14          0.358871                    download slice 
                        concat slices  1792395348.401145 1792395348.401167                22                15                16          0.000022                    concat slices  
                          save arrays  1792395348.401190 1792395348.974603            573413                17                22          0.573413                      save arrays  
    num_timestamp 18 auto-offset from tmin 1792395314727872 2026-10-19T15:35:14.727
                                TOTAL: 16131557941580467 16131557975826736          34246269                83                97


    ]sreport.desc_runprof
    [sreport.desc_ranges ranges : (9, 5, )
    .sreport.desc_ranges.descTable  ( ta,tb : range begin,end timestamps expressed as seconds from first timestamp, ab: (tb-ta) )
    [NP::descTable_ {ranges} no-DT (9, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1792395565.381901 1792395565.391286              9385                 0                 2          0.009385                              init 
                             load_geom 1792395565.391286 1792395568.739588           3348302                 2                 3          3.348302                         load_geom 
                           upload_geom 1792395568.739635 1792395569.068797            329162                 4                 5          0.329162                       upload_geom 
                         slice_genstep 1792395569.069144 1792395569.069181                37                 9                10          0.000037                     slice_genstep 
                  upload genstep slice 1792395569.069188 1792395569.105105             35917                11                12          0.035917              upload genstep slice 
                        simulate slice 1792395569.105105 1792395600.402010          31296905                12                13         31.296905                    simulate slice 
                        download slice 1792395600.402010 1792395600.765469            363459                13                14          0.363459                    download slice 
                        concat slices  1792395600.765556 1792395600.765580                24                15                16          0.000024                    concat slices  
                          save arrays  1792395600.765604 1792395607.572834           6807230                17                22          6.807230                      save arrays  
    num_timestamp 18 auto-offset from tmin 1792395565381901 2026-10-19T15:39:25.381
                                TOTAL: 16131560208689429 16131560250879850          42190421                83                97

     0 :                ta : ta
     1 :                tb : tb
     2 :                ab : ab





Enabling SER - cuts a third of the time::


    ]sreport.desc_runprof
    [sreport.desc_ranges ranges : (9, 5, )
    .sreport.desc_ranges.descTable  ( ta,tb : range begin,end timestamps expressed as seconds from first timestamp, ab: (tb-ta) )
    [NP::descTable_ {ranges} no-DT (9, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1792395736.001683 1792395736.010607              8924                 0                 2          0.008924                              init 
                             load_geom 1792395736.010607 1792395738.804113           2793506                 2                 3          2.793506                         load_geom 
                           upload_geom 1792395738.804155 1792395739.137216            333061                 4                 5          0.333061                       upload_geom 
                         slice_genstep 1792395739.137531 1792395739.137567                36                 9                10          0.000036                     slice_genstep 
                  upload genstep slice 1792395739.137573 1792395739.173158             35585                11                12          0.035585              upload genstep slice 
                        simulate slice 1792395739.173158 1792395761.172138          21998980                12                13         21.998980                    simulate slice 
                        download slice 1792395761.172138 1792395761.528129            355991                13                14          0.355991                    download slice 
                        concat slices  1792395761.528198 1792395761.528220                22                15                16          0.000022                    concat slices  
                          save arrays  1792395761.528244 1792395768.226710           6698466                17                22          6.698466                      save arrays  
    num_timestamp 18 auto-offset from tmin 1792395736001683 2026-10-19T15:42:16.001
                                TOTAL: 16131561712493287 16131561744717858          32224571                83                97


Check reproducibility::

    [sreport.desc_ranges ranges : (9, 5, )
    .sreport.desc_ranges.descTable  ( ta,tb : range begin,end timestamps expressed as seconds from first timestamp, ab: (tb-ta) )
    [NP::descTable_ {ranges} no-DT (9, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1792395945.725984 1792395945.735997             10013                 0                 2          0.010013                              init 
                             load_geom 1792395945.735997 1792395947.545850           1809853                 2                 3          1.809853                         load_geom 
                           upload_geom 1792395947.545892 1792395947.874361            328469                 4                 5          0.328469                       upload_geom 
                         slice_genstep 1792395947.874689 1792395947.874727                38                 9                10          0.000038                     slice_genstep 
                  upload genstep slice 1792395947.874734 1792395947.910399             35665                11                12          0.035665              upload genstep slice 
                        simulate slice 1792395947.910399 1792395969.925805          22015406                12                13         22.015406                    simulate slice 
                        download slice 1792395969.925805 1792395970.282517            356712                13                14          0.356712                    download slice 
                        concat slices  1792395970.282590 1792395970.282614                24                15                16          0.000024                    concat slices  
                          save arrays  1792395970.282637 1792395976.995215           6712578                17                22          6.712578                      save arrays  
    num_timestamp 18 auto-offset from tmin 1792395945725984 2026-10-19T15:45:45.725
                                TOTAL: 16131563593158727 16131563624427485          31268758                83                97

     0 :                ta : ta



After adding hintMethod:2 but before trying it::

    [sreport.desc_ranges ranges : (9, 5, )
    .sreport.desc_ranges.descTable  ( ta,tb : range begin,end timestamps expressed as seconds from first timestamp, ab: (tb-ta) )
    [NP::descTable_ {ranges} no-DT (9, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1792403553.687116 1792403553.697434             10318                 0                 2          0.010318                              init 
                             load_geom 1792403553.697434 1792403555.519887           1822453                 2                 3          1.822453                         load_geom 
                           upload_geom 1792403555.519925 1792403562.371342           6851417                 4                 5          6.851417                       upload_geom 
                         slice_genstep 1792403562.371707 1792403562.371748                41                 9                10          0.000041                     slice_genstep 
                  upload genstep slice 1792403562.371755 1792403562.407504             35749                11                12          0.035749              upload genstep slice 
                        simulate slice 1792403562.407504 1792403584.388512          21981008                12                13         21.981008                    simulate slice 
                        download slice 1792403584.388512 1792403584.982603            594091                13                14          0.594091                    download slice 
                        concat slices  1792403584.982658 1792403584.982680                22                15                16          0.000022                    concat slices  
                          save arrays  1792403584.982701 1792403591.487044           6504343                17                22          6.504343                      save arrays  
    num_timestamp 18 auto-offset from tmin 1792403553687116 2026-10-19T17:52:33.687
                                TOTAL: 16131632104409312 16131632142208754          37799442                83                97

     0 :                ta : ta


Now try::

    [Params_Helper::set_SER
     SER                        : hintMethod:2,coherenceHintBits:4,bounceInterval:4
     hintMethod                 : 2
     bounceInterval             : 4
     coherenceHintBits          : 4
    ]Params_Helper::set_SER




    [sreport.desc_ranges ranges : (9, 5, )
    .sreport.desc_ranges.descTable  ( ta,tb : range begin,end timestamps expressed as seconds from first timestamp, ab: (tb-ta) )
    [NP::descTable_ {ranges} no-DT (9, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1792403683.642459 1792403683.652998             10539                 0                 2          0.010539                              init 
                             load_geom 1792403683.652998 1792403685.439370           1786372                 2                 3          1.786372                         load_geom 
                           upload_geom 1792403685.439409 1792403685.771612            332203                 4                 5          0.332203                       upload_geom 
                         slice_genstep 1792403685.771943 1792403685.771980                37                 9                10          0.000037                     slice_genstep 
                  upload genstep slice 1792403685.771986 1792403685.807708             35722                11                12          0.035722              upload genstep slice 
                        simulate slice 1792403685.807708 1792403707.601964          21794256                12                13         21.794256                    simulate slice 
                        download slice 1792403707.601964 1792403708.195047            593083                13                14          0.593083                    download slice 
                        concat slices  1792403708.195109 1792403708.195132                23                15                16          0.000023                    concat slices  
                          save arrays  1792403708.195155 1792403714.663062           6467907                17                22          6.467907                      save arrays  
    num_timestamp 18 auto-offset from tmin 1792403683642459 2026-10-19T17:54:43.642
                                TOTAL: 16131633234078731 16131633265098873          31020142                83                97









::

    2026-10-19 15:42:19.173 INFO  [226252] [CSGOptiX::prepareParamSimulate_SER@1061] [CSGOptiX::prepareParamSimulate_SER
     SEventConfig::kSER         : OPTICKS_SER
     SEventConfig::_SERDefault  : coherenceHintBits:0,bounceInterval:0
     SEventConfig::SER()        : coherenceHintBits:4,bounceInterval:4
     bounceInterval             : 4
     coherenceHintBits          : 4
    ]CSGOptiX::prepareParamSimulate_SER









::

                     OPTICKS_SER : hintMethod:0,coherenceHintBits:4,bounceInterval:1  


    [sreport.desc_ranges ranges : (9, 5, )
    .sreport.desc_ranges.descTable  ( ta,tb : range begin,end timestamps expressed as seconds from first timestamp, ab: (tb-ta) )
    [NP::descTable_ {ranges} no-DT (9, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1792411929.994416 1792411930.004949             10533                 0                 2          0.010533                              init 
                             load_geom 1792411930.004949 1792411931.773797           1768848                 2                 3          1.768848                         load_geom 
                           upload_geom 1792411931.773834 1792411940.230068           8456234                 4                 5          8.456234                       upload_geom 
                         slice_genstep 1792411940.230444 1792411940.230484                40                 9                10          0.000040                     slice_genstep 
                  upload genstep slice 1792411940.230490 1792411940.266175             35685                11                12          0.035685              upload genstep slice 
                        simulate slice 1792411940.266175 1792411961.950564          21684389                12                13         21.684389                    simulate slice 
                        download slice 1792411961.950564 1792411962.346294            395730                13                14          0.395730                    download slice 
                        concat slices  1792411962.346362 1792411962.346385                23                15                16          0.000023                    concat slices  
                          save arrays  1792411962.346410 1792411972.168704           9822294                17                22          9.822294                      save arrays  
    num_timestamp 18 auto-offset from tmin 1792411929994416 2026-10-19T20:12:09.994
                                TOTAL: 16131707499143644 16131707541317420          42173776                83                97

     0 :                ta : ta


















