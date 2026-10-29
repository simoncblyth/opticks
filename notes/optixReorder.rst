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
8. DONE : S4MultiLayrStack.h with simple scomplex.h scomplex_select.h - to avoid heavy thrust/complex.h compiled into kernel
9. DONE : try termination hint reordering
10. DONE : resolved issue of 10s 1.2G hit.npy save times - by moving writes onto SSD /tmp not HDD /data1
11. TODO : look at impact of varying bit layout
12. try to use nsight tools to measure warp metrics, reread "Mitigating..."
13. test on RTX 6000

Overview
----------


::

    In [2]: 35/13
    Out[2]: 2.6923076923076925

Combined benefits of the below approaching a factor of 3 reduction in simulation kernel time

1. Release build
2. SER reordering
3. switch to optixir



Compare best WITH_SER with NOT:WITH_SER : 1.82x
------------------------------------------------

::

    In [1]: 12.628433/23.035775
    Out[1]: 0.5482096000677207

    In [2]: 1/(12.628433/23.035775)
    Out[2]: 1.8241198254763678




sanity check NOT:WITH_SER
--------------------------

::

    [lo] A[blyth@localhost CSGOptiX]$ cxs_min_ser.sh report | grep "simulate slice $"
    ( 0)                     simulate slice 1793268148.223157 1793268170.389169          22166012                12                13         22.166012                    simulate slice 
    ( 1)                     simulate slice 1793268171.485140 1793268193.934673          22449533                38                39         22.449533                    simulate slice 
    ( 2)                     simulate slice 1793268195.039519 1793268217.718920          22679401                64                65         22.679401                    simulate slice 
    ( 3)                     simulate slice 1793268218.812790 1793268241.877454          23064664                90                91         23.064664                    simulate slice 
    ( 4)                     simulate slice 1793268242.969544 1793268266.272539          23302995               116               117         23.302995                    simulate slice 
    ( 5)                     simulate slice 1793268267.371063 1793268290.611551          23240488               142               143         23.240488                    simulate slice 
    ( 6)                     simulate slice 1793268291.699866 1793268314.925814          23225948               168               169         23.225948                    simulate slice 
    ( 7)                     simulate slice 1793268316.010911 1793268339.045953          23035042               194               195         23.035042                    simulate slice 
    ( 8)                     simulate slice 1793268340.147889 1793268363.183664          23035775               220               221         23.035775                    simulate slice 
    ( 9)                     simulate slice 1793268364.266982 1793268387.366305          23099323               246               247         23.099323                    simulate slice 
    (10)                     simulate slice 1793268388.453580 1793268411.532179          23078599               272               273         23.078599                    simulate slice 
    (11)                     simulate slice 1793268412.623535 1793268435.695701          23072166               298               299         23.072166                    simulate slice 
    (12)                     simulate slice 1793268436.772588 1793268459.919680          23147092               324               325         23.147092                    simulate slice 
    (13)                     simulate slice 1793268461.000871 1793268484.057798          23056927               350               351         23.056927                    simulate slice 
    (14)                     simulate slice 1793268485.160674 1793268508.478501          23317827               376               377         23.317827                    simulate slice 
    (15)                     simulate slice 1793268509.560858 1793268532.691689          23130831               402               403         23.130831                    simulate slice 
    (16)                     simulate slice 1793268533.754837 1793268556.898354          23143517               428               429         23.143517                    simulate slice 
    [lo] A[blyth@localhost CSGOptiX]$ 



cxs_min_ser.sh Release
----------------------

::

       bool done = command == BREAK ;

        if( params.SER_bounceInterval > 0u && (bounce % params.SER_bounceInterval) == 0)
        {
            unsigned dn = done ? 0u : 1u ; 
            unsigned m1 = ctx.s.index.x & 0x1f ;
            unsigned m2 = ctx.s.index.y & 0x1f ;
            unsigned f2 = ctx.p.flagidx_2bit() ; 
            unsigned f4 = ctx.p.flagidx_4bit() ;
            unsigned r2 = ctx.p.ir2<4>(900.0f) ;

            switch(params.SER_hintMethod)
            {
                case  0: optixReorder()                                               ; break ;  // 14.312304
                case  1: optixReorder(  dn                                 ,  1 )     ; break ;  // 13.589810
                case  2: optixReorder(  f2                                 ,  2 )     ; break ;  // 13.267087
                case  3: optixReorder(  ( dn << 2 ) | f2                   ,  3 )     ; break ;  // 13.444653
                case  4: optixReorder(  f4                                 ,  4 )     ; break ;  // 13.437059
                case  5: optixReorder(  ( dn << 4 ) | f4                   ,  5 )     ; break ;  // 13.404350

                case  6: optixReorder(  m1                                 ,  5 )     ; break ;  // 14.005246
                case  7: optixReorder(  m2                                 ,  5 )     ; break ;  // 13.641412

                case  8: optixReorder( ( m2 << 2 ) | f2                    ,  7 )     ; break ;  // 12.850961 
                case  9: optixReorder( ( m2 << 4 ) | f4                    ,  9 )     ; break ;  // 12.658098

                case 10: optixReorder( ( r2 << 4 ) | f4                    ,  8 )     ; break ;  // 13.509211 
                case 11: optixReorder( ( m2 << 4 ) | r2                    ,  9 )     ; break ;  // 13.742220

                case 12: optixReorder( ( dn << 7 ) | ( m2 << 2 ) | f2      ,  8 )     ; break ;  // 12.801995
                case 13: optixReorder( ( dn << 7 ) | ( f2 << 5 ) | m2      ,  8 )     ; break ;  // 12.744048

                case 14: optixReorder( ( dn << 9 ) | ( m2 << 4 ) | f4      , 10 )     ; break ;  // 12.628433
                case 15: optixReorder( ( dn << 9 ) | ( f4 << 5 ) | m2      , 10 )     ; break ;  // 12.633959

                case 16:                                                              ; break ;  // 21.195825
            }
        }





        [lo] A[blyth@localhost opticks]$ cxs_min_ser.sh report | grep "simulate slice $"
        ( 0)                     simulate slice 1793269705.521660 1793269719.953723          14432063                12                13         14.432063                    simulate slice 
        ( 1)                     simulate slice 1793269721.233192 1793269735.136011          13902819                38                39         13.902819                    simulate slice 
        ( 2)                     simulate slice 1793269736.354053 1793269749.833198          13479145                64                65         13.479145                    simulate slice 
        ( 3)                     simulate slice 1793269751.074203 1793269764.675591          13601388                90                91         13.601388                    simulate slice 
        ( 4)                     simulate slice 1793269766.012738 1793269779.810964          13798226               116               117         13.798226                    simulate slice 
        ( 5)                     simulate slice 1793269781.225251 1793269794.938912          13713661               142               143         13.713661                    simulate slice 
        ( 6)                     simulate slice 1793269796.294529 1793269810.391225          14096696               168               169         14.096696                    simulate slice 
        ( 7)                     simulate slice 1793269811.754998 1793269825.619299          13864301               194               195         13.864301                    simulate slice 
        ( 8)                     simulate slice 1793269826.953359 1793269839.990940          13037581               220               221         13.037581                    simulate slice 
        ( 9)                     simulate slice 1793269841.190911 1793269854.137304          12946393               246               247         12.946393                    simulate slice 
        (10)                     simulate slice 1793269855.358768 1793269869.122315          13763547               272               273         13.763547                    simulate slice 
        (11)                     simulate slice 1793269870.438429 1793269884.367572          13929143               298               299         13.929143                    simulate slice 
        (12)                     simulate slice 1793269885.664580 1793269898.567530          12902950               324               325         12.902950                    simulate slice 
        (13)                     simulate slice 1793269899.740069 1793269912.947997          13207928               350               351         13.207928                    simulate slice 
        (14)                     simulate slice 1793269914.346973 1793269927.111920          12764947               376               377         12.764947                    simulate slice 
        (15)                     simulate slice 1793269928.258943 1793269941.117037          12858094               402               403         12.858094                    simulate slice 
        (16)                     simulate slice 1793269942.237185 1793269963.526634          21289449               428               429         21.289449                    simulate slice 
        [lo] A[blyth@localhost opticks]$ 


        [lo] A[blyth@localhost CSGOptiX]$ cxs_min_ser.sh report | grep "simulate slice $"
        ( 0)                     simulate slice 1793266753.541499 1793266767.853803          14312304                12                13         14.312304                    simulate slice 
        ( 1)                     simulate slice 1793266769.009203 1793266782.599013          13589810                38                39         13.589810                    simulate slice 
        ( 2)                     simulate slice 1793266783.691114 1793266796.958201          13267087                64                65         13.267087                    simulate slice 
        ( 3)                     simulate slice 1793266798.060141 1793266811.504794          13444653                90                91         13.444653                    simulate slice 
        ( 4)                     simulate slice 1793266812.595688 1793266826.032747          13437059               116               117         13.437059                    simulate slice 
        ( 5)                     simulate slice 1793266827.209072 1793266840.613422          13404350               142               143         13.404350                    simulate slice 
        ( 6)                     simulate slice 1793266841.770302 1793266855.775548          14005246               168               169         14.005246                    simulate slice 
        ( 7)                     simulate slice 1793266857.043279 1793266870.684691          13641412               194               195         13.641412                    simulate slice 
        ( 8)                     simulate slice 1793266871.809569 1793266884.660530          12850961               220               221         12.850961                    simulate slice 
        ( 9)                     simulate slice 1793266885.757039 1793266898.415137          12658098               246               247         12.658098                    simulate slice 
        (10)                     simulate slice 1793266899.603172 1793266913.112383          13509211               272               273         13.509211                    simulate slice 
        (11)                     simulate slice 1793266914.274086 1793266928.016306          13742220               298               299         13.742220                    simulate slice 
        (12)                     simulate slice 1793266929.137587 1793266941.939582          12801995               324               325         12.801995                    simulate slice 
        (13)                     simulate slice 1793266942.998122 1793266955.742170          12744048               350               351         12.744048                    simulate slice 
        (14)                     simulate slice 1793266956.800013 1793266969.428446          12628433               376               377         12.628433                    simulate slice 
        (15)                     simulate slice 1793266970.482423 1793266983.116382          12633959               402               403         12.633959                    simulate slice 
        (16)                     simulate slice 1793266983.854131 1793267005.049956          21195825               428               429         21.195825                    simulate slice 
        [lo] A[blyth@localhost CSGOptiX]$ 






















FIXED : Why is "save arrays" taking 10s ? That is download + saving
------------------------------------------------------------------------

sreport.h::

     39     static constexpr const char* RANGES = R"(
     40         SEvt__Init_RUN_META:CSGFoundry__Load_HEAD                     ## init
     41         CSGFoundry__Load_HEAD:CSGFoundry__Load_TAIL                   ## load_geom
     42         CSGOptiX__Create_HEAD:CSGOptiX__Create_TAIL                   ## upload_geom
     43         A%0.3d_QSim__simulate_HEAD:A%0.3d_QSim__simulate_LBEG         ## slice_genstep
     44         A%0.3d_QSim__simulate_PRUP:A%0.3d_QSim__simulate_PREL         ## upload genstep slice
     45         A%0.3d_QSim__simulate_PREL:A%0.3d_QSim__simulate_POST         ## simulate slice
     46         A%0.3d_QSim__simulate_POST:A%0.3d_QSim__simulate_DOWN         ## download slice
     47         A%0.3d_QSim__simulate_LEND:A%0.3d_QSim__simulate_PCAT         ## concat slices
     48         A%0.3d_QSim__simulate_BRES:A%0.3d_QSim__simulate_TAIL         ## save arrays
     49        )" ;


::

     584     int64_t t_BRES  = SProf::Add("QSim__simulate_BRES", counts.c_str() );
     585     if(reset_) reset(eventID) ;   // reset calles SEvt::endOfEvent
     586 
     587     int64_t t_TAIL  = SProf::Add("QSim__simulate_TAIL");
     588 

::

     855 void QSim::reset(int eventID)
     856 {
     857     SProf::Add("QSim__reset_HEAD");
     858     qev->clear();  
     859     sev->endOfEvent(eventID);
     860     LOG_IF(info, SEvt::LIFECYCLE) << "] eventID " << eventID ;
     861     SProf::Add("QSim__reset_TAIL");
     862 }






SCANNING hintMethod
--------------------

* with the M100 test - find that adding more bits in the coherenceHit improves the SER benefit


Reach 40% SER benefit::

    In [1]: 12.901081/21.541187
    Out[1]: 0.5989029759594956

    In [2]: 1-(12.901081/21.541187)
    Out[2]: 0.4010970240405044

    In [4]: 1/(12.901081/21.541187)
    Out[4]: 1.669719537455815




::

    cat << EOS > $ser_path
    hintMethod:0,coherenceHintBits:-1,bounceInterval:1
    hintMethod:1,coherenceHintBits:-1,bounceInterval:1
    hintMethod:2,coherenceHintBits:-1,bounceInterval:1
    hintMethod:3,coherenceHintBits:-1,bounceInterval:1
    hintMethod:4,coherenceHintBits:-1,bounceInterval:1
    hintMethod:5,coherenceHintBits:-1,bounceInterval:1
    hintMethod:6,coherenceHintBits:-1,bounceInterval:1
    hintMethod:7,coherenceHintBits:-1,bounceInterval:1
    hintMethod:8,coherenceHintBits:-1,bounceInterval:1
    hintMethod:9,coherenceHintBits:-1,bounceInterval:1
    hintMethod:10,coherenceHintBits:-1,bounceInterval:1
    EOS



     610         bool done = command == BREAK ;
     611 
     612         if( params.SER_bounceInterval > 0u && (bounce % params.SER_bounceInterval) == 0)
     613         {
     614             switch(params.SER_hintMethod)
     615             {
     616                 case 0:  optixReorder();                                                          break ;
     617                 case 1:  optixReorder( done ? 0 : 1                                     , 1 );    break ;
     618                 case 2:  optixReorder(   ctx.p.flagidx_2bit()                           , 2 );    break ;
     619                 case 3:  optixReorder( ( ctx.p.flagidx_2bit() << 1 ) | ( done ? 0 : 1 ) , 3 );    break ;
     620                 case 4:  optixReorder( ctx.p.flagidx_4bit()                             , 4 );    break ;
     621                 case 5:  optixReorder( ( ctx.p.flagidx_4bit() << 1 ) | ( done ? 0 : 1 ) , 5 );    break ;
     622                 case 6:  optixReorder( ctx.s.index.x                                    , 5 );    break ;  // m1 : 5 bits enough for 32 material indices 
     623                 case 7:  optixReorder( ctx.s.index.y                                    , 5 );    break ;  // m2
     624                 case 8:  optixReorder( ( ctx.s.index.y << 2 ) | ctx.p.flagidx_2bit()    , 7 );    break ;  // m2{5b} with f2{2b}     
     625                 case 9:  optixReorder( ( ctx.s.index.y << 4 ) | ctx.p.flagidx_4bit()    , 9 );    break ;  // m2{5b} with f4{4b}     
     626                 case 10:                                                                     ;    break ;
     627             }
     628         }
     629 
     630         bounce++;

::

    ( 0)                     simulate slice 1793172191.825844 1793172206.420733          14594889                12                13         14.594889                    simulate slice 
    ( 1)                     simulate slice 1793172216.959138 1793172230.922548          13963410                26                27         13.963410                    simulate slice 
    ( 2)                     simulate slice 1793172241.208282 1793172254.779090          13570808                40                41         13.570808                    simulate slice 
    ( 3)                     simulate slice 1793172265.027173 1793172278.686607          13659434                54                55         13.659434                    simulate slice 
    ( 4)                     simulate slice 1793172288.916373 1793172302.495525          13579152                68                69         13.579152                    simulate slice 
    ( 5)                     simulate slice 1793172312.728479 1793172326.413541          13685062                82                83         13.685062                    simulate slice 
    ( 6)                     simulate slice 1793172336.830569 1793172351.068001          14237432                96                97         14.237432   m1-not-so-good   simulate slice 
    ( 7)                     simulate slice 1793172361.900088 1793172375.791978          13891890               110               111         13.891890                    simulate slice 
    ( 8)                     simulate slice 1793172386.113483 1793172399.115475          13001992               124               125         13.001992                    simulate slice 
    ( 9)                     simulate slice 1793172409.364066 1793172422.265147          12901081               138               139         12.901081   m2{5b}|f4{4b}    simulate slice 
    (10)                     simulate slice 1793172432.379792 1793172453.920979          21541187               152               153         21.541187                    simulate slice 




    [NP::descTable_ {ranges} no-DT (69, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                       init 1793172189.983280 1793172189.991358              8078                 0                 2          0.008078                              init 
                                  load_geom 1793172189.991358 1793172191.537579           1546221                 2                 3          1.546221                         load_geom 
                                upload_geom 1793172191.537618 1793172191.802955            265337                 4                 5          0.265337                       upload_geom 
                              slice_genstep 1793172191.803235 1793172191.803258                23                 9                10          0.000023                     slice_genstep 
                       upload genstep slice 1793172191.803264 1793172191.825844             22580                11                12          0.022580              upload genstep slice 
    ( 0)                     simulate slice 1793172191.825844 1793172206.420733          14594889                12                13         14.594889                    simulate slice 
                             download slice 1793172206.420733 1793172206.784796            364063                13                14          0.364063                    download slice 
                             concat slices  1793172206.784847 1793172206.784863                16                15                16          0.000016                    concat slices  
                               save arrays  1793172206.784880 1793172216.956425          10171545                17                21         10.171545                      save arrays  
                              slice_genstep 1793172216.956525 1793172216.956538                13                23                24          0.000013                     slice_genstep 
                       upload genstep slice 1793172216.956544 1793172216.959138              2594                25                26          0.002594              upload genstep slice 
    ( 1)                     simulate slice 1793172216.959138 1793172230.922548          13963410                26                27         13.963410                    simulate slice 
                             download slice 1793172230.922548 1793172231.285817            363269                27                28          0.363269                    download slice 
                             concat slices  1793172231.285865 1793172231.285880                15                29                30          0.000015                    concat slices  
                               save arrays  1793172231.285891 1793172241.205596           9919705                31                35          9.919705                      save arrays  
                              slice_genstep 1793172241.205689 1793172241.205701                12                37                38          0.000012                     slice_genstep 
                       upload genstep slice 1793172241.205707 1793172241.208282              2575                39                40          0.002575              upload genstep slice 
    ( 2)                     simulate slice 1793172241.208282 1793172254.779090          13570808                40                41         13.570808                    simulate slice 
                             download slice 1793172254.779090 1793172255.142260            363170                41                42          0.363170                    download slice 
                             concat slices  1793172255.142308 1793172255.142324                16                43                44          0.000016                    concat slices  
                               save arrays  1793172255.142335 1793172265.024423           9882088                45                49          9.882088                      save arrays  
                              slice_genstep 1793172265.024521 1793172265.024534                13                51                52          0.000013                     slice_genstep 
                       upload genstep slice 1793172265.024539 1793172265.027173              2634                53                54          0.002634              upload genstep slice 
    ( 3)                     simulate slice 1793172265.027173 1793172278.686607          13659434                54                55         13.659434                    simulate slice 
                             download slice 1793172278.686607 1793172279.048807            362200                55                56          0.362200                    download slice 
                             concat slices  1793172279.048861 1793172279.048877                16                57                58          0.000016                    concat slices  
                               save arrays  1793172279.048888 1793172288.910862           9861974                59                63          9.861974                      save arrays  
                              slice_genstep 1793172288.913766 1793172288.913779                13                65                66          0.000013                     slice_genstep 
                       upload genstep slice 1793172288.913785 1793172288.916373              2588                67                68          0.002588              upload genstep slice 
    ( 4)                     simulate slice 1793172288.916373 1793172302.495525          13579152                68                69         13.579152                    simulate slice 
                             download slice 1793172302.495525 1793172302.860449            364924                69                70          0.364924                    download slice 
                             concat slices  1793172302.860497 1793172302.860514                17                71                72          0.000017                    concat slices  
                               save arrays  1793172302.860525 1793172312.721645           9861120                73                77          9.861120                      save arrays  
                              slice_genstep 1793172312.725886 1793172312.725900                14                79                80          0.000014                     slice_genstep 
                       upload genstep slice 1793172312.725906 1793172312.728479              2573                81                82          0.002573              upload genstep slice 
    ( 5)                     simulate slice 1793172312.728479 1793172326.413541          13685062                82                83         13.685062                    simulate slice 
                             download slice 1793172326.413541 1793172326.775412            361871                83                84          0.361871                    download slice 
                             concat slices  1793172326.775467 1793172326.775483                16                85                86          0.000016                    concat slices  
                               save arrays  1793172326.775494 1793172336.827804          10052310                87                91         10.052310                      save arrays  
                              slice_genstep 1793172336.827962 1793172336.827974                12                93                94          0.000012                     slice_genstep 
                       upload genstep slice 1793172336.827980 1793172336.830569              2589                95                96          0.002589              upload genstep slice 
    ( 6)                     simulate slice 1793172336.830569 1793172351.068001          14237432                96                97         14.237432                    simulate slice 
                             download slice 1793172351.068001 1793172351.440610            372609                97                98          0.372609                    download slice 
                             concat slices  1793172351.440663 1793172351.440679                16                99               100          0.000016                    concat slices  
                               save arrays  1793172351.440690 1793172361.885205          10444515               101               105         10.444515                      save arrays  
                              slice_genstep 1793172361.897482 1793172361.897497                15               107               108          0.000015                     slice_genstep 
                       upload genstep slice 1793172361.897503 1793172361.900088              2585               109               110          0.002585              upload genstep slice 
    ( 7)                     simulate slice 1793172361.900088 1793172375.791978          13891890               110               111         13.891890                    simulate slice 
                             download slice 1793172375.791978 1793172376.154920            362942               111               112          0.362942                    download slice 
                             concat slices  1793172376.154978 1793172376.154994                16               113               114          0.000016                    concat slices  
                               save arrays  1793172376.155004 1793172386.110351           9955347               115               119          9.955347                      save arrays  
                              slice_genstep 1793172386.110888 1793172386.110900                12               121               122          0.000012                     slice_genstep 
                       upload genstep slice 1793172386.110905 1793172386.113483              2578               123               124          0.002578              upload genstep slice 
    ( 8)                     simulate slice 1793172386.113483 1793172399.115475          13001992               124               125         13.001992                    simulate slice 
                             download slice 1793172399.115475 1793172399.477869            362394               125               126          0.362394                    download slice 
                             concat slices  1793172399.477912 1793172399.477928                16               127               128          0.000016                    concat slices  
                               save arrays  1793172399.477941 1793172409.361296           9883355               129               133          9.883355                      save arrays  
                              slice_genstep 1793172409.361437 1793172409.361450                13               135               136          0.000013                     slice_genstep 
                       upload genstep slice 1793172409.361455 1793172409.364066              2611               137               138          0.002611              upload genstep slice 
    ( 9)                     simulate slice 1793172409.364066 1793172422.265147          12901081               138               139         12.901081                    simulate slice 
                             download slice 1793172422.265147 1793172422.627696            362549               139               140          0.362549                    download slice 
                             concat slices  1793172422.627744 1793172422.627759                15               141               142          0.000015                    concat slices  
                               save arrays  1793172422.627770 1793172432.364354           9736584               143               147          9.736584                      save arrays  
                              slice_genstep 1793172432.377189 1793172432.377205                16               149               150          0.000016                     slice_genstep 
                       upload genstep slice 1793172432.377210 1793172432.379792              2582               151               152          0.002582              upload genstep slice 
    (10)                     simulate slice 1793172432.379792 1793172453.920979          21541187               152               153         21.541187                    simulate slice 
                             download slice 1793172453.920979 1793172454.294446            373467               153               154          0.373467                    download slice 
                             concat slices  1793172454.294502 1793172454.294518                16               155               156          0.000016                    concat slices  
                               save arrays  1793172454.294528 1793172464.701626          10407098               157               162         10.407098                      save arrays  
    num_timestamp 138 auto-offset from tmin 1793172189983280 2026-10-28T15:23:09.983
                                TOTAL: 123728889714842135 123728889989526027         274683892              5473              5577









SCANNING num coherenceBits - DO THEY MAKE ANY DIFF
------------------------------------------------------


* DONE sphoton::flagidx_2bit
* TRY: changing the order of bits

::

        bool done = command == BREAK ;

        if( params.SER_bounceInterval > 0u && (bounce % params.SER_bounceInterval) == 0)
        {
            switch(params.SER_coherenceHintBits)
            {
                case 0:  optixReorder();                                                      break ;
                case 1:  optixReorder( done ? 0 : 1                                 , 1 );    break ;
                case 2:  optixReorder( ctx.p.flagidx_2bit()                         , 2 );    break ;
                case 3:  optixReorder( ctx.p.flagidx_2bit() << 1 | ( done ? 0 : 1 ) , 3 );    break ;
                case 4:  optixReorder( ctx.p.flagidx_4bit()                         , 4 );    break ;
                case 5:  optixReorder( ctx.p.flagidx_4bit() << 1 | ( done ? 0 : 1 ) , 5 );    break ;
            }
        }

        bounce++;



     610         bool done = command == BREAK ;
     612         if( params.SER_bounceInterval > 0u && (bounce % params.SER_bounceInterval) == 0)
     613         {
     614             switch(params.SER_coherenceHintBits)
     615             {
     616                 case 0:  optixReorder();                                                      break ;
     617                 case 1:  optixReorder( done ? 0 : 1                                 , 1 );    break ;
     618                 case 2:  optixReorder( ctx.p.flagidx_2bit()                         , 2 );    break ;
     619                 case 3:  optixReorder( ctx.p.flagidx_2bit() << 1 | ( done ? 0 : 1 ) , 3 );    break ;
     620                 case 4:  optixReorder( ctx.p.flagidx_4bit()                         , 4 );    break ;  // <<<
     621                 case 5:  optixReorder( ctx.p.flagidx_4bit() << 1 | ( done ? 0 : 1 ) , 5 );    break ;
     622                 case 6:  optixReorder( ctx.s.index.x                                , 5 );    break ;  // m1 : 5 bits enough for 32 material indices
     623                 case 7:  optixReorder( ctx.s.index.y                                , 5 );    break ;  // m2                    <<  0.2s better than m1
     624                 case 8:                                                                  ;    break ;
     625             }
     626         }
     628         bounce++;

     479    cat << EOS > $ser_path
     480 hintMethod:0,coherenceHintBits:0,bounceInterval:1
     481 hintMethod:0,coherenceHintBits:1,bounceInterval:1
     482 hintMethod:0,coherenceHintBits:2,bounceInterval:1
     483 hintMethod:0,coherenceHintBits:3,bounceInterval:1
     484 hintMethod:0,coherenceHintBits:4,bounceInterval:1
     485 hintMethod:0,coherenceHintBits:5,bounceInterval:1
     486 hintMethod:0,coherenceHintBits:6,bounceInterval:1
     487 hintMethod:0,coherenceHintBits:7,bounceInterval:1
     488 hintMethod:0,coherenceHintBits:8,bounceInterval:0
     489 EOS


    ( 0)                     simulate slice 1793169513.815574 1793169528.373307          14557733                12                13         14.557733                    simulate slice 
    ( 1)                     simulate slice 1793169538.934389 1793169552.895820          13961431                26                27         13.961431                    simulate slice 
    ( 2)                     simulate slice 1793169563.343337 1793169576.932923          13589586                40                41         13.589586                    simulate slice 
    ( 3)                     simulate slice 1793169587.552811 1793169601.193442          13640631                54                55         13.640631                    simulate slice 
    ( 4)                     simulate slice 1793169611.516494 1793169625.014141          13497647                68                69         13.497647                    simulate slice 
    ( 5)                     simulate slice 1793169635.894745 1793169649.437927          13543182                82                83         13.543182                    simulate slice 
    ( 6)                     simulate slice 1793169659.680816 1793169673.791933          14111117                96                97         14.111117                    simulate slice 
    ( 7)                     simulate slice 1793169684.060563 1793169697.876648          13816085               110               111         13.816085                    simulate slice 
    ( 8)                     simulate slice 1793169708.025501 1793169729.459942          21434441               124               125         21.434441                    simulate slice 

    [NP::descTable_ {ranges} no-DT (57, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                       init 1793169504.410161 1793169504.417708              7547                 0                 2          0.007547                              init 
                                  load_geom 1793169504.417708 1793169505.947087           1529379                 2                 3          1.529379                         load_geom 
                                upload_geom 1793169505.947120 1793169513.793237           7846117                 4                 5          7.846117                       upload_geom 
                              slice_genstep 1793169513.793530 1793169513.793555                25                 9                10          0.000025                     slice_genstep 
                       upload genstep slice 1793169513.793561 1793169513.815574             22013                11                12          0.022013              upload genstep slice 
    ( 0)                     simulate slice 1793169513.815574 1793169528.373307          14557733                12                13         14.557733                    simulate slice 
                             download slice 1793169528.373307 1793169528.732806            359499                13                14          0.359499                    download slice 
                             concat slices  1793169528.732864 1793169528.732882                18                15                16          0.000018                    concat slices  
                               save arrays  1793169528.732899 1793169538.931659          10198760                17                21         10.198760                      save arrays  
                              slice_genstep 1793169538.931768 1793169538.931782                14                23                24          0.000014                     slice_genstep 
                       upload genstep slice 1793169538.931789 1793169538.934389              2600                25                26          0.002600              upload genstep slice 
    ( 1)                     simulate slice 1793169538.934389 1793169552.895820          13961431                26                27         13.961431                    simulate slice 
                             download slice 1793169552.895820 1793169553.260029            364209                27                28          0.364209                    download slice 
                             concat slices  1793169553.260085 1793169553.260103                18                29                30          0.000018                    concat slices  
                               save arrays  1793169553.260115 1793169563.340617          10080502                31                35         10.080502                      save arrays  
                              slice_genstep 1793169563.340726 1793169563.340740                14                37                38          0.000014                     slice_genstep 
                       upload genstep slice 1793169563.340746 1793169563.343337              2591                39                40          0.002591              upload genstep slice 
    ( 2)                     simulate slice 1793169563.343337 1793169576.932923          13589586                40                41         13.589586                    simulate slice 
                             download slice 1793169576.932923 1793169577.296574            363651                41                42          0.363651                    download slice 
                             concat slices  1793169577.296625 1793169577.296640                15                43                44          0.000015                    concat slices  
                               save arrays  1793169577.296652 1793169587.549993          10253341                45                49         10.253341                      save arrays  
                              slice_genstep 1793169587.550103 1793169587.550116                13                51                52          0.000013                     slice_genstep 
                       upload genstep slice 1793169587.550122 1793169587.552811              2689                53                54          0.002689              upload genstep slice 
    ( 3)                     simulate slice 1793169587.552811 1793169601.193442          13640631                54                55         13.640631                    simulate slice 
                             download slice 1793169601.193442 1793169601.551823            358381                55                56          0.358381                    download slice 
                             concat slices  1793169601.551873 1793169601.551889                16                57                58          0.000016                    concat slices  
                               save arrays  1793169601.551900 1793169611.513765           9961865                59                63          9.961865                      save arrays  
                              slice_genstep 1793169611.513893 1793169611.513905                12                65                66          0.000012                     slice_genstep 
                       upload genstep slice 1793169611.513911 1793169611.516494              2583                67                68          0.002583              upload genstep slice 
    ( 4)                     simulate slice 1793169611.516494 1793169625.014141          13497647                68                69         13.497647                    simulate slice 
                             download slice 1793169625.014141 1793169625.372938            358797                69                70          0.358797                    download slice 
                             concat slices  1793169625.372987 1793169625.373004                17                71                72          0.000017                    concat slices  
                               save arrays  1793169625.373014 1793169635.892011          10518997                73                77         10.518997                      save arrays  
                              slice_genstep 1793169635.892139 1793169635.892151                12                79                80          0.000012                     slice_genstep 
                       upload genstep slice 1793169635.892157 1793169635.894745              2588                81                82          0.002588              upload genstep slice 
    ( 5)                     simulate slice 1793169635.894745 1793169649.437927          13543182                82                83         13.543182                    simulate slice 
                             download slice 1793169649.437927 1793169649.802279            364352                83                84          0.364352                    download slice 
                             concat slices  1793169649.802341 1793169649.802358                17                85                86          0.000017                    concat slices  
                               save arrays  1793169649.802370 1793169659.674394           9872024                87                91          9.872024                      save arrays  
                              slice_genstep 1793169659.678200 1793169659.678214                14                93                94          0.000014                     slice_genstep 
                       upload genstep slice 1793169659.678220 1793169659.680816              2596                95                96          0.002596              upload genstep slice 
    ( 6)                     simulate slice 1793169659.680816 1793169673.791933          14111117                96                97         14.111117                    simulate slice 
                             download slice 1793169673.791933 1793169674.152168            360235                97                98          0.360235                    download slice 
                             concat slices  1793169674.152224 1793169674.152241                17                99               100          0.000017                    concat slices  
                               save arrays  1793169674.152252 1793169684.057784           9905532               101               105          9.905532                      save arrays  
                              slice_genstep 1793169684.057940 1793169684.057954                14               107               108          0.000014                     slice_genstep 
                       upload genstep slice 1793169684.057961 1793169684.060563              2602               109               110          0.002602              upload genstep slice 
    ( 7)                     simulate slice 1793169684.060563 1793169697.876648          13816085               110               111         13.816085                    simulate slice 
                             download slice 1793169697.876648 1793169698.238607            361959               111               112          0.361959                    download slice 
                             concat slices  1793169698.238668 1793169698.238683                15               113               114          0.000015                    concat slices  
                               save arrays  1793169698.238694 1793169708.022756           9784062               115               119          9.784062                      save arrays  
                              slice_genstep 1793169708.022896 1793169708.022908                12               121               122          0.000012                     slice_genstep 
                       upload genstep slice 1793169708.022914 1793169708.025501              2587               123               124          0.002587              upload genstep slice 
    ( 8)                     simulate slice 1793169708.025501 1793169729.459942          21434441               124               125         21.434441                    simulate slice 
                             download slice 1793169729.459942 1793169729.818438            358496               125               126          0.358496                    download slice 
                             concat slices  1793169729.818494 1793169729.818509                15               127               128          0.000015                    concat slices  
                               save arrays  1793169729.818524 1793169739.563149           9744625               129               134          9.744625                      save arrays  
    num_timestamp 114 auto-offset from tmin 1793169504410161 2026-10-28T14:38:24.410
                                TOTAL: 102210667934590459 102210668169737769         235147310              3723              3809

     0 :                ta : ta
     1 :                tb : tb







Release : optixTraverse + optixInvoke - with .optixir - standard heavy thrust/complex - termination reorder with bounceInterval scan
-------------------------------------------------------------------------------------------------------------------------------------




::

    elif [ "$TEST" == "ser_medium_scan" ]; then


       ser_path=/tmp/$USER/opticks/$name/SER.txt
       mkdir -p $(dirname $ser_path)
       export OPTICKS_SER=filepath:$ser_path
       cat << EOS > $ser_path
    hintMethod:0,coherenceHintBits:0,bounceInterval:0
    hintMethod:0,coherenceHintBits:0,bounceInterval:1
    hintMethod:0,coherenceHintBits:0,bounceInterval:2
    hintMethod:0,coherenceHintBits:0,bounceInterval:3
    hintMethod:0,coherenceHintBits:0,bounceInterval:4
    hintMethod:0,coherenceHintBits:0,bounceInterval:5
    hintMethod:0,coherenceHintBits:0,bounceInterval:6
    hintMethod:0,coherenceHintBits:0,bounceInterval:7
    hintMethod:0,coherenceHintBits:0,bounceInterval:8
    hintMethod:0,coherenceHintBits:0,bounceInterval:9
    hintMethod:0,coherenceHintBits:0,bounceInterval:10
    EOS
       ## with current kernel only bounceInterval does anything

       opticks_num_event=$(wc -l < $ser_path)
       opticks_num_photon=M100x$opticks_num_event
       opticks_num_genstep=1x$opticks_num_event
       opticks_running_mode=SRM_TORCH
       opticks_max_slot=M100  ## avoid creating big buffer
       opticks_max_bounce=63


The speedup from SER degrades as reorder done less often (0 means no-SER) giving 21s::

                   0    simulate slice 1793107944.378724 1793107965.473117          21094393                12                13         21.094393                    simulate slice 
                   1    simulate slice 1793107975.622035 1793107989.188265          13566230                26                27         13.566230                    simulate slice 
                   2    simulate slice 1793107999.408205 1793108013.296492          13888287                40                41         13.888287                    simulate slice 
                   3    simulate slice 1793108023.440617 1793108037.728056          14287439                54                55         14.287439                    simulate slice 
                   4    simulate slice 1793108048.533057 1793108063.257681          14724624                68                69         14.724624                    simulate slice 
                   5    simulate slice 1793108073.568223 1793108088.780908          15212685                82                83         15.212685                    simulate slice 
                   6    simulate slice 1793108089.701279 1793108105.180172          15478893                96                97         15.478893                    simulate slice 
                   7    simulate slice 1793108106.117995 1793108121.784039          15666044               110               111         15.666044                    simulate slice 
                   8    simulate slice 1793108122.698125 1793108138.648831          15950706               124               125         15.950706                    simulate slice 
                   9    simulate slice 1793108139.598410 1793108155.784146          16185736               138               139         16.185736                    simulate slice 
                  10    simulate slice 1793108156.740277 1793108173.204442          16464165               152               153         16.464165                    simulate slice 



A bit more than a third reduction in simulation time from termination reorder::

    In [2]: 13.566230/21.094393
    Out[2]: 0.6431201883837093

    In [3]: 1-(13.566230/21.094393)
    Out[3]: 0.3568798116162907






::

    [sreport.desc_ranges ranges : (69, 5, )
    .sreport.desc_ranges.descTable  ( ta,tb : range begin,end timestamps expressed as seconds from first timestamp, ab: (tb-ta) )
    [NP::descTable_ {ranges} no-DT (69, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1793107942.559808 1793107942.568041              8233                 0                 2          0.008233                              init 
                             load_geom 1793107942.568041 1793107944.109613           1541572                 2                 3          1.541572                         load_geom 
                           upload_geom 1793107944.109645 1793107944.355758            246113                 4                 5          0.246113                       upload_geom 
                         slice_genstep 1793107944.356017 1793107944.356038                21                 9                10          0.000021                     slice_genstep 
                  upload genstep slice 1793107944.356045 1793107944.378724             22679                11                12          0.022679              upload genstep slice 
                        simulate slice 1793107944.378724 1793107965.473117          21094393                12                13         21.094393                    simulate slice 
                        download slice 1793107965.473117 1793107965.834819            361702                13                14          0.361702                    download slice 
                        concat slices  1793107965.834870 1793107965.834888                18                15                16          0.000018                    concat slices  
                          save arrays  1793107965.834904 1793107975.619332           9784428                17                21          9.784428                      save arrays  
                         slice_genstep 1793107975.619428 1793107975.619440                12                23                24          0.000012                     slice_genstep 
                  upload genstep slice 1793107975.619446 1793107975.622035              2589                25                26          0.002589              upload genstep slice 
                        simulate slice 1793107975.622035 1793107989.188265          13566230                26                27         13.566230                    simulate slice 
                        download slice 1793107989.188265 1793107989.551669            363404                27                28          0.363404                    download slice 
                        concat slices  1793107989.551720 1793107989.551736                16                29                30          0.000016                    concat slices  
                          save arrays  1793107989.551747 1793107999.405226           9853479                31                35          9.853479                      save arrays  
                         slice_genstep 1793107999.405320 1793107999.405334                14                37                38          0.000014                     slice_genstep 
                  upload genstep slice 1793107999.405339 1793107999.408205              2866                39                40          0.002866              upload genstep slice 
                        simulate slice 1793107999.408205 1793108013.296492          13888287                40                41         13.888287                    simulate slice 
                        download slice 1793108013.296492 1793108013.657394            360902                41                42          0.360902                    download slice 
                        concat slices  1793108013.657440 1793108013.657455                15                43                44          0.000015                    concat slices  
                          save arrays  1793108013.657465 1793108023.437893           9780428                45                49          9.780428                      save arrays  
                         slice_genstep 1793108023.437995 1793108023.438009                14                51                52          0.000014                     slice_genstep 
                  upload genstep slice 1793108023.438015 1793108023.440617              2602                53                54          0.002602              upload genstep slice 
                        simulate slice 1793108023.440617 1793108037.728056          14287439                54                55         14.287439                    simulate slice 
                        download slice 1793108037.728056 1793108038.089431            361375                55                56          0.361375                    download slice 
                        concat slices  1793108038.089483 1793108038.089498                15                57                58          0.000015                    concat slices  
                          save arrays  1793108038.089509 1793108048.530325          10440816                59                63         10.440816                      save arrays  
                         slice_genstep 1793108048.530441 1793108048.530455                14                65                66          0.000014                     slice_genstep 
                  upload genstep slice 1793108048.530461 1793108048.533057              2596                67                68          0.002596              upload genstep slice 
                        simulate slice 1793108048.533057 1793108063.257681          14724624                68                69         14.724624                    simulate slice 
                        download slice 1793108063.257681 1793108063.622575            364894                69                70          0.364894                    download slice 
                        concat slices  1793108063.622623 1793108063.622639                16                71                72          0.000016                    concat slices  
                          save arrays  1793108063.622650 1793108073.561786           9939136                73                77          9.939136                      save arrays  
                         slice_genstep 1793108073.565616 1793108073.565629                13                79                80          0.000013                     slice_genstep 
                  upload genstep slice 1793108073.565635 1793108073.568223              2588                81                82          0.002588              upload genstep slice 
                        simulate slice 1793108073.568223 1793108088.780908          15212685                82                83         15.212685                    simulate slice 
                        download slice 1793108088.780908 1793108089.141870            360962                83                84          0.360962                    download slice 
                        concat slices  1793108089.141916 1793108089.141931                15                85                86          0.000015                    concat slices  
                          save arrays  1793108089.141942 1793108089.698574            556632                87                91          0.556632                      save arrays  
                         slice_genstep 1793108089.698691 1793108089.698703                12                93                94          0.000012                     slice_genstep 
                  upload genstep slice 1793108089.698709 1793108089.701279              2570                95                96          0.002570              upload genstep slice 
                        simulate slice 1793108089.701279 1793108105.180172          15478893                96                97         15.478893                    simulate slice 
                        download slice 1793108105.180172 1793108105.542908            362736                97                98          0.362736                    download slice 
                        concat slices  1793108105.542957 1793108105.542976                19                99               100          0.000019                    concat slices  
                          save arrays  1793108105.542987 1793108106.098270            555283               101               105          0.555283                      save arrays  
                         slice_genstep 1793108106.115390 1793108106.115406                16               107               108          0.000016                     slice_genstep 
                  upload genstep slice 1793108106.115413 1793108106.117995              2582               109               110          0.002582              upload genstep slice 
                        simulate slice 1793108106.117995 1793108121.784039          15666044               110               111         15.666044                    simulate slice 
                        download slice 1793108121.784039 1793108122.150438            366399               111               112          0.366399                    download slice 
                        concat slices  1793108122.150482 1793108122.150497                15               113               114          0.000015                    concat slices  
                          save arrays  1793108122.150508 1793108122.695403            544895               115               119          0.544895                      save arrays  
                         slice_genstep 1793108122.695529 1793108122.695543                14               121               122          0.000014                     slice_genstep 
                  upload genstep slice 1793108122.695549 1793108122.698125              2576               123               124          0.002576              upload genstep slice 
                        simulate slice 1793108122.698125 1793108138.648831          15950706               124               125         15.950706                    simulate slice 
                        download slice 1793108138.648831 1793108139.010467            361636               125               126          0.361636                    download slice 
                        concat slices  1793108139.010520 1793108139.010537                17               127               128          0.000017                    concat slices  
                          save arrays  1793108139.010552 1793108139.575354            564802               129               133          0.564802                      save arrays  
                         slice_genstep 1793108139.595806 1793108139.595819                13               135               136          0.000013                     slice_genstep 
                  upload genstep slice 1793108139.595825 1793108139.598410              2585               137               138          0.002585              upload genstep slice 
                        simulate slice 1793108139.598410 1793108155.784146          16185736               138               139         16.185736                    simulate slice 
                        download slice 1793108155.784146 1793108156.145293            361147               139               140          0.361147                    download slice 
                        concat slices  1793108156.145335 1793108156.145349                14               141               142          0.000014                    concat slices  
                          save arrays  1793108156.145360 1793108156.710099            564739               143               147          0.564739                      save arrays  
                         slice_genstep 1793108156.737682 1793108156.737697                15               149               150          0.000015                     slice_genstep 
                  upload genstep slice 1793108156.737703 1793108156.740277              2574               151               152          0.002574              upload genstep slice 
                        simulate slice 1793108156.740277 1793108173.204442          16464165               152               153         16.464165                    simulate slice 
                        download slice 1793108173.204442 1793108173.565193            360751               153               154          0.360751                    download slice 
                        concat slices  1793108173.565246 1793108173.565261                15               155               156          0.000015                    concat slices  
                          save arrays  1793108173.565272 1793108174.097330            532058               157               162          0.532058                      save arrays  
    num_timestamp 138 auto-offset from tmin 1793107942559808 2026-10-27T21:32:22.559
                                TOTAL: 123724456433512133 123724456664978997         231466864              5473              5577






Release : optixTraverse + optixReorder + optixInvoke - with .optixir - standard heavy thrust/complex - termination reorder with hit object correction
-------------------------------------------------------------------------------------------------------------------------------------------------------

::

    /**
    corrected_MakeHitObject
    -------------------------

    This to check if reordering based on corrected
    spatial info from the photon simulation pos and mom
    makes any difference to performance enhancement of reorder.

    **/

    static __forceinline__ __device__ void corrected_MakeHitObject(
            OptixTraversableHandle handle,
            float3                 ray_origin,
            float3                 ray_direction,
            float                  tmin
            )
    {

        const float rayTime = 0.0f ;
        OptixRayFlags rayFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT ;   // OPTIX_RAY_FLAG_NONE
     
        OptixTraverseData traverseData;
        optixHitObjectGetTraverseData(&traverseData);

        unsigned int numTransforms = optixHitObjectGetTransformListSize();
        OptixTraversableHandle transforms[4];
        for(unsigned int i = 0; i < numTransforms && i < 4; ++i) {
            transforms[i] = optixHitObjectGetTransformListHandle(i);
        }    

        optixMakeHitObject(
            handle, 
            ray_origin, 
            ray_direction, 
            tmin, 
            rayTime, 
            rayFlags, 
            traverseData, 
            transforms, 
            numTransforms
        );   
    }








::

    ]sreport.desc_runprof
    [sreport.desc_ranges ranges : (33, 5, )
    .sreport.desc_ranges.descTable  ( ta,tb : range begin,end timestamps expressed as seconds from first timestamp, ab: (tb-ta) )
    [NP::descTable_ {ranges} no-DT (33, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1793017424.411792 1793017424.420405              8613                 0                 2          0.008613                              init 
                             load_geom 1793017424.420405 1793017425.992858           1572453                 2                 3          1.572453                         load_geom 
                           upload_geom 1793017425.992891 1793017429.762017           3769126                 4                 5          3.769126                       upload_geom 
                         slice_genstep 1793017429.762318 1793017429.762341                23                 9                10          0.000023                     slice_genstep 
                  upload genstep slice 1793017429.762346 1793017429.784218             21872                11                12          0.021872              upload genstep slice 
                        simulate slice 1793017429.784218 1793017443.367259          13583041                12                13         13.583041                    simulate slice 
                        download slice 1793017443.367259 1793017443.730828            363569                13                14          0.363569                    download slice 
                        concat slices  1793017443.730905 1793017443.730922                17                15                16          0.000017                    concat slices  
                          save arrays  1793017443.730940 1793017453.677590           9946650                17                21          9.946650                      save arrays  
                         slice_genstep 1793017453.677707 1793017453.677720                13                23                24          0.000013                     slice_genstep 
                  upload genstep slice 1793017453.677726 1793017453.680321              2595                25                26          0.002595              upload genstep slice 
                        simulate slice 1793017453.680321 1793017467.479601          13799280                26                27         13.799280                    simulate slice 
                        download slice 1793017467.479601 1793017467.855659            376058                27                28          0.376058                    download slice 
                        concat slices  1793017467.855721 1793017467.855740                19                29                30          0.000019                    concat slices  
                          save arrays  1793017467.855753 1793017477.670984           9815231                31                35          9.815231                      save arrays  
                         slice_genstep 1793017477.681949 1793017477.681962                13                37                38          0.000013                     slice_genstep 
                  upload genstep slice 1793017477.681967 1793017477.684564              2597                39                40          0.002597              upload genstep slice 
                        simulate slice 1793017477.684564 1793017491.532624          13848060                40                41         13.848060                    simulate slice 
                        download slice 1793017491.532624 1793017491.894962            362338                41                42          0.362338                    download slice 
                        concat slices  1793017491.895020 1793017491.895037                17                43                44          0.000017                    concat slices  
                          save arrays  1793017491.895048 1793017502.295344          10400296                45                49         10.400296                      save arrays  
                         slice_genstep 1793017502.309727 1793017502.309741                14                51                52          0.000014                     slice_genstep 
                  upload genstep slice 1793017502.309747 1793017502.312342              2595                53                54          0.002595              upload genstep slice 
                        simulate slice 1793017502.312342 1793017516.277924          13965582                54                55         13.965582                    simulate slice 
                        download slice 1793017516.277924 1793017516.640885            362961                55                56          0.362961                    download slice 
                        concat slices  1793017516.640945 1793017516.640961                16                57                58          0.000016                    concat slices  
                          save arrays  1793017516.640972 1793017526.601176           9960204                59                63          9.960204                      save arrays  
                         slice_genstep 1793017526.601294 1793017526.601309                15                65                66          0.000015                     slice_genstep 
                  upload genstep slice 1793017526.601315 1793017526.603909              2594                67                68          0.002594              upload genstep slice 
                        simulate slice 1793017526.603909 1793017547.785022          21181113                68                69         21.181113                    simulate slice 
                        download slice 1793017547.785022 1793017548.146423            361401                69                70          0.361401                    download slice 
                        concat slices  1793017548.146478 1793017548.146494                16                71                72          0.000016                    concat slices  
                          save arrays  1793017548.146505 1793017557.787123           9640618                73                78          9.640618                      save arrays  
    num_timestamp 66 auto-offset from tmin 1793017424411792 2026-10-26T20:23:44.411
                                TOTAL: 59169576847937255 59169576981286265         133349010              1231              1281











Release : optixTraverse + optixReorder + optixInvoke - with .optixir - standard heavy thrust/complex - just termination reorder
----------------------------------------------------------------------------------------------------------------------------------------


::

    sreport.desc_ranges.descTable  ( ta,tb : range begin,end timestamps expressed as seconds from first timestamp, ab: (tb-ta) )
    [NP::descTable_ {ranges} no-DT (33, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1792761672.640297 1792761672.647713              7416                 0                 2          0.007416                              init 
                             load_geom 1792761672.647713 1792761674.081474           1433761                 2                 3          1.433761                         load_geom 
                           upload_geom 1792761674.081506 1792761677.976361           3894855                 4                 5          3.894855                       upload_geom 
                         slice_genstep 1792761677.976643 1792761677.976664                21                 9                10          0.000021                     slice_genstep 
                  upload genstep slice 1792761677.976670 1792761677.998934             22264                11                12          0.022264              upload genstep slice 
                        simulate slice 1792761677.998934 1792761691.405027          13406093                12                13         13.406093                    simulate slice 
                        download slice 1792761691.405027 1792761691.766455            361428                13                14          0.361428                    download slice 
                        concat slices  1792761691.766508 1792761691.766525                17                15                16          0.000017                    concat slices  
                          save arrays  1792761691.766541 1792761701.362661           9596120                17                21          9.596120                      save arrays  
                         slice_genstep 1792761701.362768 1792761701.362781                13                23                24          0.000013                     slice_genstep 
                  upload genstep slice 1792761701.362786 1792761701.365301              2515                25                26          0.002515              upload genstep slice 
                        simulate slice 1792761701.365301 1792761715.730978          14365677                26                27         14.365677                    simulate slice 
                        download slice 1792761715.730978 1792761716.093480            362502                27                28          0.362502                    download slice 
                        concat slices  1792761716.093526 1792761716.093542                16                29                30          0.000016                    concat slices  
                          save arrays  1792761716.093552 1792761726.521266          10427714                31                35         10.427714                      save arrays  
                         slice_genstep 1792761726.533850 1792761726.533863                13                37                38          0.000013                     slice_genstep 
                  upload genstep slice 1792761726.533868 1792761726.536573              2705                39                40          0.002705              upload genstep slice 
                        simulate slice 1792761726.536573 1792761740.542422          14005849                40                41         14.005849                    simulate slice 
                        download slice 1792761740.542422 1792761740.905507            363085                41                42          0.363085                    download slice 
                        concat slices  1792761740.905564 1792761740.905579                15                43                44          0.000015                    concat slices  
                          save arrays  1792761740.905590 1792761750.727257           9821667                45                49          9.821667                      save arrays  
                         slice_genstep 1792761750.727384 1792761750.727396                12                51                52          0.000012                     slice_genstep 
                  upload genstep slice 1792761750.727401 1792761750.729990              2589                53                54          0.002589              upload genstep slice 
                        simulate slice 1792761750.729990 1792761764.545974          13815984                54                55         13.815984                    simulate slice 
                        download slice 1792761764.545974 1792761764.906653            360679                55                56          0.360679                    download slice 
                        concat slices  1792761764.906698 1792761764.906714                16                57                58          0.000016                    concat slices  
                          save arrays  1792761764.906725 1792761774.680004           9773279                59                63          9.773279                      save arrays  
                         slice_genstep 1792761774.680127 1792761774.680141                14                65                66          0.000014                     slice_genstep 
                  upload genstep slice 1792761774.680147 1792761774.682737              2590                67                68          0.002590              upload genstep slice 
                        simulate slice 1792761774.682737 1792761795.539813          20857076                68                69         20.857076                    simulate slice 
                        download slice 1792761795.539813 1792761795.900038            360225                69                70          0.360225                    download slice 
                        concat slices  1792761795.900092 1792761795.900115                23                71                72          0.000023                    concat slices  
                          save arrays  1792761795.900125 1792761805.401593           9501468                73                78          9.501468                      save arrays  
    num_timestamp 66 auto-offset from tmin 1792761672640297 2026-10-23T21:21:12.640
                                TOTAL: 59161137040153830 59161137172901531         132747701              1231              1281

     0 :                ta : ta





Release : optixTraverse + optixReorder + optixInvoke - with .optixir - standard heavy thrust/complex - just termination reorder
----------------------------------------------------------------------------------------------------------------------------------------


* note - all have SER


::

    NP::descTable_ {ranges} no-DT (33, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1792760352.667128 1792760352.674601              7473                 0                 2          0.007473                              init 
                             load_geom 1792760352.674601 1792760354.192494           1517893                 2                 3          1.517893                         load_geom 
                           upload_geom 1792760354.192528 1792760358.043762           3851234                 4                 5          3.851234                       upload_geom 
                         slice_genstep 1792760358.044046 1792760358.044069                23                 9                10          0.000023                     slice_genstep 
                  upload genstep slice 1792760358.044075 1792760358.066091             22016                11                12          0.022016              upload genstep slice 
                        simulate slice 1792760358.066091 1792760371.486963          13420872                12                13         13.420872                    simulate slice 
                        download slice 1792760371.486963 1792760372.005899            518936                13                14          0.518936                    download slice 
                        concat slices  1792760372.005968 1792760372.005987                19                15                16          0.000019                    concat slices  
                          save arrays  1792760372.006008 1792760383.141211          11135203                17                21         11.135203                      save arrays  
                         slice_genstep 1792760383.141309 1792760383.141324                15                23                24          0.000015                     slice_genstep 
                  upload genstep slice 1792760383.141330 1792760383.143910              2580                25                26          0.002580              upload genstep slice 
                        simulate slice 1792760383.143910 1792760396.835064          13691154                26                27         13.691154                    simulate slice 
                        download slice 1792760396.835064 1792760397.340634            505570                27                28          0.505570                    download slice 
                        concat slices  1792760397.340688 1792760397.340703                15                29                30          0.000015                    concat slices  
                          save arrays  1792760397.340714 1792760407.998618          10657904                31                35         10.657904                      save arrays  
                         slice_genstep 1792760407.998718 1792760407.998734                16                37                38          0.000016                     slice_genstep 
                  upload genstep slice 1792760407.998740 1792760408.001411              2671                39                40          0.002671              upload genstep slice 
                        simulate slice 1792760408.001411 1792760421.851719          13850308                40                41         13.850308                    simulate slice 
                        download slice 1792760421.851719 1792760422.340677            488958                41                42          0.488958                    download slice 
                        concat slices  1792760422.340726 1792760422.340740                14                43                44          0.000014                    concat slices  
                          save arrays  1792760422.340751 1792760431.758511           9417760                45                49          9.417760                      save arrays  
                         slice_genstep 1792760431.758619 1792760431.758632                13                51                52          0.000013                     slice_genstep 
                  upload genstep slice 1792760431.758638 1792760431.761217              2579                53                54          0.002579              upload genstep slice 
                        simulate slice 1792760431.761217 1792760445.661472          13900255                54                55         13.900255                    simulate slice 
                        download slice 1792760445.661472 1792760446.144421            482949                55                56          0.482949                    download slice 
                        concat slices  1792760446.144488 1792760446.144504                16                57                58          0.000016                    concat slices  
                          save arrays  1792760446.144516 1792760455.770073           9625557                59                63          9.625557                      save arrays  
                         slice_genstep 1792760455.770195 1792760455.770207                12                65                66          0.000012                     slice_genstep 
                  upload genstep slice 1792760455.770212 1792760455.772788              2576                67                68          0.002576              upload genstep slice 
                        simulate slice 1792760455.772788 1792760469.687893          13915105                68                69         13.915105                    simulate slice 
                        download slice 1792760469.687893 1792760470.126428            438535                69                70          0.438535                    download slice 
                        concat slices  1792760470.126483 1792760470.126499                16                71                72          0.000016                    concat slices  
                          save arrays  1792760470.126511 1792760479.898315           9771804                73                78          9.771804                      save arrays  
    num_timestamp 66 auto-offset from tmin 1792760352667128 2026-10-23T20:59:12.667
                                TOTAL: 59161093491145520 59161093618375571         127230051              1231              1281









Release : optixTraverse + optixReorder + optixInvoke - with .optixir - and using the S4MultiLayrStack.h - AVOIDING THE HEAVY thrust/complex
-------------------------------------------------------------------------------------------------------------------------------------------

* NO SIGNIFICANT CHANGE - SO NOT WORTH VALIDATING S4MultiLayrStack.h ?

::

    [NP::descTable_ {ranges} no-DT (33, 5, )
                                                      ta                tb                ab                ia                ib          TS_DELTA
                                  init 1792743029.697723 1792743029.705430              7707                 0                 2          0.007707                              init 
                             load_geom 1792743029.705430 1792743031.167953           1462523                 2                 3          1.462523                         load_geom 
                           upload_geom 1792743031.168004 1792743034.139601           2971597                 4                 5          2.971597                       upload_geom 
                         slice_genstep 1792743034.139931 1792743034.139952                21                 9                10          0.000021                     slice_genstep 
                  upload genstep slice 1792743034.139957 1792743034.161978             22021                11                12          0.022021              upload genstep slice 
                        simulate slice 1792743034.161978 1792743048.154014          13992036                12                13         13.992036                    simulate slice 
                        download slice 1792743048.154014 1792743048.710554            556540                13                14          0.556540                    download slice 
                        concat slices  1792743048.710599 1792743048.710616                17                15                16          0.000017                    concat slices  
                          save arrays  1792743048.710634 1792743058.581405           9870771                17                21          9.870771                      save arrays  
                         slice_genstep 1792743058.581514 1792743058.581527                13                23                24          0.000013                     slice_genstep 
                  upload genstep slice 1792743058.581533 1792743058.584114              2581                25                26          0.002581              upload genstep slice 
                        simulate slice 1792743058.584114 1792743073.780145          15196031                26                27         15.196031                    simulate slice 
                        download slice 1792743073.780145 1792743074.332547            552402                27                28          0.552402                    download slice 
                        concat slices  1792743074.332584 1792743074.332598                14                29                30          0.000014                    concat slices  
                          save arrays  1792743074.332607 1792743083.168880           8836273                31                35          8.836273                      save arrays  
                         slice_genstep 1792743083.168989 1792743083.169004                15                37                38          0.000015                     slice_genstep 
                  upload genstep slice 1792743083.169009 1792743083.171588              2579                39                40          0.002579              upload genstep slice 
                        simulate slice 1792743083.171588 1792743097.344112          14172524                40                41         14.172524                    simulate slice 
                        download slice 1792743097.344112 1792743097.682538            338426                41                42          0.338426                    download slice 
                        concat slices  1792743097.682585 1792743097.682600                15                43                44          0.000015                    concat slices  
                          save arrays  1792743097.682611 1792743106.773157           9090546                45                49          9.090546                      save arrays  
                         slice_genstep 1792743106.773284 1792743106.773297                13                51                52          0.000013                     slice_genstep 
                  upload genstep slice 1792743106.773303 1792743106.775884              2581                53                54          0.002581              upload genstep slice 
                        simulate slice 1792743106.775884 1792743121.013910          14238026                54                55         14.238026                    simulate slice 
                        download slice 1792743121.013910 1792743121.352553            338643                55                56          0.338643                    download slice 
                        concat slices  1792743121.352603 1792743121.352619                16                57                58          0.000016                    concat slices  
                          save arrays  1792743121.352629 1792743130.487427           9134798                59                63          9.134798                      save arrays  
                         slice_genstep 1792743130.487547 1792743130.487560                13                65                66          0.000013                     slice_genstep 
                  upload genstep slice 1792743130.487565 1792743130.490136              2571                67                68          0.002571              upload genstep slice 
                        simulate slice 1792743130.490136 1792743152.664502          22174366                68                69         22.174366                    simulate slice 
                        download slice 1792743152.664502 1792743153.004176            339674                69                70          0.339674                    download slice 
                        concat slices  1792743153.004215 1792743153.004231                16                71                72          0.000016                    concat slices  
                          save arrays  1792743153.004242 1792743161.931228           8926986                73                78          8.926986                      save arrays  
    num_timestamp 66 auto-offset from tmin 1792743029697723 2026-10-23T16:10:29.697
                                TOTAL: 59160521813179481 59160521945411836         132232355              1231              1281






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



