cxs_min_save_arrays_very_slow_in_Debug_build
==============================================


HUH compare back with Debug, notice that "save arrays" is factor of 10-20x slower for the Debug build ?

* from ~/o/notes/optixReorder.rst

SER scanning::

    TEST=ser_medium_scan cxs_min.sh

Compare Release build sreport::

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


With Debug build::

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


hit array 1.3G
----------------

::

    A[blyth@localhost A000]$ du -h hit.npy 
    1.3G	hit.npy


Is "save arrays" really all array saving ?
----------------------------------------------

sreport::RANGES::

       A%0.3d_QSim__simulate_BRES:A%0.3d_QSim__simulate_TAIL         ## save arrays

::

     584     int64_t t_BRES  = SProf::Add("QSim__simulate_BRES", counts.c_str() );
     585     if(reset_) reset(eventID) ;   // reset calles SEvt::endOfEvent
     587     int64_t t_TAIL  = SProf::Add("QSim__simulate_TAIL");



     840 /**
     841 QSim::reset
     842 ------------
     843 
     844 When *QSim::simulate* is called with argument *reset:true* the
     845 *QSim::reset* method is called which invokes SEvt::endOfEvent in order
     846 to clean up the SEvt. Normally that would be done after saving
     847 any Opticks configured arrays.
     848 
     849 When *QSim::simulate* is called with argument *reset:false*
     850 the *QSim::reset* method must be called separately in order to avoid a memory leak.
     851 Using *reset:false* is typically done in order to keep arrays alive longer
     852 to enable copying from the gathered arrays into non-Opticks collections.
     853 
     854 **/
     855 void QSim::reset(int eventID)
     856 {
     857     SProf::Add("QSim__reset_HEAD");
     858     qev->clear();
     859     sev->endOfEvent(eventID);
     860     LOG_IF(info, SEvt::LIFECYCLE) << "] eventID " << eventID ;
     861     SProf::Add("QSim__reset_TAIL");
     862 }


     265 /**
     266 QEvt::clear
     267 --------------
     268 
     269 This is called from QSim::reset
     270 The former omission of gs deletion was reported by Ilker Parmaksiz.
     271 
     272 **/
     273 
     274 void QEvt::clear()
     275 {
     276     delete gs ;
     277     gs = nullptr ;
     278 }



