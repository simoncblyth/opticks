U4RecorderTest_cf_CXRaindropTest : low stat non-aligned comparisons of U4RecorderTest and CXRaindropTest : finding big issues
================================================================================================================================

* from :doc:`U4RecorderTest-shakedown`

Doing this with::

    cd ~/opticks/u4/tests
    ./U4RecorderTest_ab.sh 



cx missing seq 
------------------

::

    35 const char* SEventConfig::_CompMaskDefault = SComp::ALL_ ;

    038 struct SYSRAP_API SComp
     39 {
     40     static constexpr const char* ALL_ = "genstep,photon,record,rec,seq,seed,hit,simtrace,domain,inphoton" ;
     41     static constexpr const char* UNDEFINED_ = "undefined" ;
     42     static constexpr const char* GENSTEP_   = "genstep" ;


::

    2022-06-14 22:18:07.758 INFO  [386951] [SEvt::save@944] DefaultDir /tmp/blyth/opticks/GeoChain/BoxedSphere/CXRaindropTest
    2022-06-14 22:18:07.758 INFO  [386951] [SEvt::save@970]  dir /tmp/blyth/opticks/GeoChain/BoxedSphere/CXRaindropTest
    2022-06-14 22:18:07.758 INFO  [386951] [QEvent::getPhoton@345] [ evt.num_photon 10 p.sstr (10, 4, 4, ) evt.photon 0x7f75ec000000
    2022-06-14 22:18:07.758 INFO  [386951] [QEvent::getPhoton@348] ] evt.num_photon 10
    2022-06-14 22:18:07.758 INFO  [386951] [QEvent::getRecord@404]  evt.num_record 100
    2022-06-14 22:18:07.758 INFO  [386951] [QEvent::getRec@411]  getRec called when there is no such array, use SEventConfig::SetCompMask to avoid 
    2022-06-14 22:18:07.758 INFO  [386951] [QEvent::getSeq@388]  getSeq called when there is no such array, use SEventConfig::SetCompMask to avoid 
    2022-06-14 22:18:07.761 INFO  [386951] [QEvent::getHit@479]  evt.photon 0x7f75ec000000 evt.num_photon 10 evt.num_hit 0 selector.hitmask 64 SEventConfig::HitMask 64 SEventConfig::HitMaskLabel SD
    2022-06-14 22:18:07.761 INFO  [386951] [QEvent::getSimtrace@370]  getSimtrace called when there is no such array, use SEventConfig::SetCompMask to avoid 
    2022-06-14 22:18:07.761 INFO  [386951] [SEvt::save@974] SEvt::descComponent
     SEventConfig::CompMaskLabel genstep,photon,record,rec,seq,seed,hit,simtrace,domain,inphoton
                     hit                    - 
                    seed               (10, ) 
                 genstep          (1, 6, 4, )       SEventConfig::MaxGenstep             1000000
                  photon         (10, 4, 4, )        SEventConfig::MaxPhoton             3000000
                  record     (10, 10, 4, 4, )        SEventConfig::MaxRecord                  10
                     rec                    -           SEventConfig::MaxRec                   0
                     seq                    -           SEventConfig::MaxSeq                   0
                  domain          (2, 4, 4, ) 
                simtrace                    - 

    2022-06-14 22:18:07.761 INFO  [386951] [SEvt::save@975] NPFold::desc
                                 genstep.npy : (1, 6, 4, )
                                  photon.npy : (10, 4, 4, )
                                  record.npy : (10, 10, 4, 4, )
                                    seed.npy : (10, )
                                  domain.npy : (2, 4, 4, )
                                inphoton.npy : (10, 4, 4, )


::

    249 bool QEvent::hasSeq() const    { return evt->seq != nullptr ; }

    377 void QEvent::getSeq(NP* seq) const
    378 {
    379     if(!hasSeq()) return ;
    380     LOG(LEVEL) << "[ evt.num_seq " << evt->num_seq << " seq.sstr " << seq->sstr() << " evt.seq " << evt->seq ;
    381     assert( seq->has_shape(evt->num_seq, 2) );
    382     QU::copy_device_to_host<sseq>( (sseq*)seq->bytes(), evt->seq, evt->num_seq );
    383     LOG(LEVEL) << "] evt.num_seq " << evt->num_seq  ;
    384 }



The defaults are all zero for debug records::

     17 int SEventConfig::_MaxRecordDefault = 0 ;
     18 int SEventConfig::_MaxRecDefault = 0 ;
     19 int SEventConfig::_MaxSeqDefault = 0 ;

And cxs_raindrop.sh only upped that for RECORD, now added REC and SEQ::

     91 unset GEOM                     # MUST unset GEOM for CSGFoundry::Load_ to load OPTICKS_KEY basis geometry 
     92 export OPTICKS_MAX_RECORD=10   # change from default of 0, see sysrap/SEventConfig.cc
     93 export OPTICKS_MAX_SEQ=10
     94 export OPTICKS_MAX_REC=10
     95 

From U4RecorderTest::

    164     unsigned max_bounce = 9 ;
    165     SEventConfig::SetMaxBounce(max_bounce);
    166     SEventConfig::SetMaxRecord(max_bounce+1);
    167     SEventConfig::SetMaxRec(max_bounce+1);
    168     SEventConfig::SetMaxSeq(max_bounce+1);


Consolidate to make it easier for debug executables to use same config settings::

    void SEventConfig::SetStandardFullDebug() // static
    {
        unsigned max_bounce = 9 ; 
        SEventConfig::SetMaxBounce(max_bounce); 
        SEventConfig::SetMaxRecord(max_bounce+1); 
        SEventConfig::SetMaxRec(max_bounce+1); 
        SEventConfig::SetMaxSeq(max_bounce+1); 
    }





cx 2/10 with nan polz
-------------------------

::

    In [59]: a.photon[:,2]                                                                                                                                                      
    Out[59]: 
    array([[ -0.544,   0.009,  -0.839, 440.   ],
           [    nan,     nan,     nan, 440.   ],
           [  0.179,  -0.457,  -0.871, 440.   ],
           [  0.757,   0.404,   0.513, 440.   ],
           [    nan,     nan,     nan, 440.   ],
           [  0.923,  -0.337,   0.183, 440.   ],
           [  0.965,   0.   ,  -0.262, 440.   ],
           [ -0.753,   0.   ,   0.658, 440.   ],
           [ -0.566,   0.   ,   0.824, 440.   ],
           [ -0.256,  -0.948,   0.19 , 440.   ]], dtype=float32)




    In [43]: a.record[1,:4]                                                                                                                                                     
    Out[43]: 
    array([[[  -0.217,   -0.975,    0.058,    0.2  ],
            [  -0.217,   -0.975,    0.058,    1.   ],
            [  -0.258,    0.   ,   -0.966,  440.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[ -10.831,  -48.727,    2.889,    0.426],
            [  -0.217,   -0.975,    0.058,    0.   ],
            [     nan,      nan,      nan,  440.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[ -22.228, -100.   ,    5.93 ,    0.602],
            [  -0.217,   -0.975,    0.058,    0.   ],
            [     nan,      nan,      nan,  440.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[   0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]]], dtype=float32)

    In [58]: a.record[4,:4]                                                                                                                                                     
    Out[58]: 
    array([[[  -0.456,    0.237,   -0.858,    0.5  ],
            [  -0.456,    0.237,   -0.858,    1.   ],
            [   0.883,    0.   ,   -0.469,  440.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[ -22.789,   11.855,  -42.896,    0.726],
            [  -0.456,    0.237,   -0.858,    0.   ],
            [     nan,      nan,      nan,  440.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[ -53.126,   27.637, -100.   ,    0.948],
            [  -0.456,    0.237,   -0.858,    0.   ],
            [     nan,      nan,      nan,  440.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[   0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ],
            [   0.   ,    0.   ,    0.   ,    0.   ]]], dtype=float32)





::

    a.base:/tmp/blyth/opticks/GeoChain/BoxedSphere/CXRaindropTest

      : a.genstep                                          :            (1, 6, 4) : 0:27:47.278953 
      : a.seed                                             :                (10,) : 0:27:47.276945 
      : a.record_meta                                      :                    1 : 0:27:47.277345 
      : a.NPFold_meta                                      :                    2 : 0:27:47.280458 
      : a.record                                           :       (10, 10, 4, 4) : 0:27:47.277733 
      : a.domain                                           :            (2, 4, 4) : 0:27:47.279858 
      : a.inphoton                                         :           (10, 4, 4) : 0:27:47.278531 
      : a.NPFold_index                                     :                    6 : 0:27:47.281013 
      : a.photon                                           :           (10, 4, 4) : 0:27:47.278158 
      : a.domain_meta                                      :                    2 : 0:27:47.279315 

     min_stamp : 2022-06-14 15:47:50.299234 
     max_stamp : 2022-06-14 15:47:50.303302 
     dif_stamp : 0:00:00.004068 
     age_stamp : 0:27:47.276945 

    In [37]: b                                                                                                                                                                  
    Out[37]: 
    b

    CMDLINE:/Users/blyth/opticks/u4/tests/U4RecorderTest_ab.py
    b.base:/tmp/blyth/opticks/U4RecorderTest

      : b.genstep                                          :            (1, 6, 4) : 0:21:56.990119 
      : b.seq                                              :              (10, 2) : 0:21:56.988098 
      : b.record_meta                                      :                    1 : 0:21:56.989270 
      : b.pho0                                             :              (10, 4) : 0:21:56.985779 
      : b.rec_meta                                         :                    1 : 0:21:56.988635 
      : b.rec                                              :       (10, 10, 2, 4) : 0:21:56.988532 
      : b.record                                           :       (10, 10, 4, 4) : 0:21:56.989174 
      : b.domain                                           :            (2, 4, 4) : 0:21:56.986951 
      : b.inphoton                                         :           (10, 4, 4) : 0:21:56.986110 
      : b.pho                                              :              (10, 4) : 0:21:56.985578 
      : b.NPFold_index                                     :                    7 : 0:21:56.990755 
      : b.photon                                           :           (10, 4, 4) : 0:21:56.989561 
      : b.gs                                               :               (1, 4) : 0:21:56.985400 
      : b.domain_meta                                      :                    2 : 0:21:56.987080 

     min_stamp : 2022-06-14 15:53:42.157865 
     max_stamp : 2022-06-14 15:53:42.163220 




post: time is off, must be different refractive index ?::

    In [25]: a.photon[:,0]                                                                                                                                                      
    Out[25]: 
    array([[-100.   ,  -31.67 ,   75.357,    0.59 ],
           [ -22.228, -100.   ,    5.93 ,    0.602],
           [-100.   ,  -75.341,   17.199,    0.781],
           [ -59.225,  -17.159,  100.   ,    0.851],
           [ -53.126,   27.637, -100.   ,    0.948],
           [  41.563,   54.208,  100.   ,    1.525],
           [ -27.109,   11.211, -100.   ,    1.107],
           [  87.27 ,  -70.573,  100.   ,    1.361],
           [ 100.   ,  -23.237,   68.731,    1.372],
           [ -67.583,   60.769,  100.   ,    1.51 ]], dtype=float32)

    In [26]: b.photon[:,0]                                                                                                                                                      
    Out[26]: 
    array([[-100.   ,  -31.67 ,   75.357,    0.689],
           [ -22.228, -100.   ,    5.93 ,    0.667],
           [-100.   ,  -75.341,   17.199,    0.876],
           [ -59.225,  -17.159,  100.   ,    0.935],
           [ -53.126,   27.637, -100.   ,    1.031],
              [ -41.563,  -54.208, -100.   ,    1.152],        OPPOSITE POS ?
           [ -27.109,   11.211, -100.   ,    1.174],
           [  87.27 ,  -70.573,  100.   ,    1.486],
           [ 100.   ,  -23.237,   68.731,    1.463],
           [ -67.583,   60.769,  100.   ,    1.616]], dtype=float32)


::

    In [33]: at
    Out[33]: array([0.59 , 0.602, 0.781, 0.851, 0.948, 1.525, 1.107, 1.361, 1.372, 1.51 ], dtype=float32)

    In [34]: bt
    Out[34]: array([0.689, 0.667, 0.876, 0.935, 1.031, 1.152, 1.174, 1.486, 1.463, 1.616], dtype=float32)

    In [35]: bt/at
    Out[35]: array([1.167, 1.108, 1.122, 1.098, 1.087, 0.755, 1.061, 1.092, 1.067, 1.07 ], dtype=float32)





