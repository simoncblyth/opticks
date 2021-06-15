pflags_ana_BT_SD_SI_zero
============================

* because EC|EX flags are currently only in pflags for OK they mess up the pflags comparison
* excluding EC|EX from pflags verifies that

* HMM: is it possible to communicate the SD/G4 cull decision from ProcessHits 
  into CRecorder to set the EC|EX pflags so can then do a fair comparison without pflags scrubbing 


jsd::

     265 G4bool junoSD_PMT_v2::ProcessHits(G4Step * step,G4TouchableHistory*)
     266 {
     ...
     453     bool de_cull = G4UniformRand() > de ;
     454 
     455 #ifdef WITH_G4OPTICKS
     456     {
     457         if(m_ce_mode == "20inch") m_PMTEfficiencyCheck->addHitRecord( pmtID, global_pos, local_pos, qe, ce, de, volname, ce_cat);
     458         // default PMTSDMgr:CollEffiMode is None, but tut_detsim.py defaullt args.ce_mode --ce-mode is 20inch
     459 
     460         G4OpticksRecorder* recorder = G4OpticksRecorder::Get();
     461         if(recorder)
     462         {
     463             recorder->ProcessHits( step, !de_cull );
     464         }
     465     }
     466 #endif
     467 
     468     if (de_cull) {
     469         return false;
     470     }   



::

    +
    +void CManager::ProcessHits( const G4Step* step, bool efficiency_collect )
    +{
    +    const G4Track* track = step->GetTrack();    
    +    bool fabricate_unlabelled = false ;
    +    CPho chit = CPhotonInfo::Get(track, fabricate_unlabelled); 
    +    LOG(LEVEL) << " chit " << chit.desc() << " efficiency_collect " << efficiency_collect ; 
    +
    +}






After temporarily scrubbing "EX|EC" from pflags, remove the zeros. But looks to be more water AB from OK::

    In [1]: ab.flg                                                                                                                                                                                          
    Out[1]: 
    ab.flg
    .       pflags_ana  cfo:sum  1:g4live:tds3gun   -1:g4live:tds3gun        c2        ab        ba 
    .                              11278     11278       374.49/29 = 12.91  (pval:0.000 prob:1.000)  
    0000                a      1653      1665    -12             0.04        0.993 +- 0.024        1.007 +- 0.025  [2 ] AB|SI
    0001              842      1345      1389    -44             0.71        0.968 +- 0.026        1.033 +- 0.028  [3 ] BT|SD|SI
    0002              882      1200      1224    -24             0.24        0.980 +- 0.028        1.020 +- 0.029  [3 ] BT|SA|SI
    0003              862       943      1020    -77             3.02        0.925 +- 0.030        1.082 +- 0.034  [4 ] BT|SD|SC|SI
    0004              8a2       680       766    -86             5.11        0.888 +- 0.034        1.126 +- 0.041  [4 ] BT|SA|SC|SI
    0005               1a       586       742   -156            18.33        0.790 +- 0.033        1.266 +- 0.046  [3 ] RE|AB|SI
    0006              852       537       662   -125            13.03        0.811 +- 0.035        1.233 +- 0.048  [4 ] BT|SD|RE|SI
    0007               2a       601       540     61             3.26        1.113 +- 0.045        0.899 +- 0.039  [3 ] SC|AB|SI
    0008              872       530       569    -39             1.38        0.931 +- 0.040        1.074 +- 0.045  [5 ] BT|SD|SC|RE|SI
    0009              892       422       522   -100            10.59        0.808 +- 0.039        1.237 +- 0.054  [4 ] BT|SA|RE|SI
    0010              8b2       384       478    -94            10.25        0.803 +- 0.041        1.245 +- 0.057  [5 ] BT|SA|SC|RE|SI
    0011               3a       431       421     10             0.12        1.024 +- 0.049        0.977 +- 0.048  [4 ] SC|RE|AB|SI
    0012              82a       419       170    249           105.26        2.465 +- 0.120        0.406 +- 0.031  [4 ] BT|SC|AB|SI
    0013              80a       386       196    190            62.03        1.969 +- 0.100        0.508 +- 0.036  [3 ] BT|AB|SI
    0014              83a       282       102    180            84.38        2.765 +- 0.165        0.362 +- 0.036  [5 ] BT|SC|RE|AB|SI
    0015              81a       202        93    109            40.27        2.172 +- 0.153        0.460 +- 0.048  [4 ] BT|RE|AB|SI
    0016                9       142       144     -2             0.01        0.986 +- 0.083        1.014 +- 0.085  [2 ] AB|CK
    0017              832        78       110    -32             5.45        0.709 +- 0.080        1.410 +- 0.134  [4 ] BT|SC|RE|SI
    0018              c32        54        47      7             0.49        1.149 +- 0.156        0.870 +- 0.127  [5 ] BT|BR|SC|RE|SI
    0019              c22        39        45     -6             0.43        0.867 +- 0.139        1.154 +- 0.172  [4 ] BT|BR|SC|SI
    .                              11278     11278       374.49/29 = 12.91  (pval:0.000 prob:1.000)  



ana/evt.py::

     575         allpflags = ox.view(np.uint32)[:,3,3]
     576         self.allpflags = allpflags
     577 
     578         self.c4 = c4
     579 
     580         all_pflags_ana = self.make_pflags_ana( self.pflags, "all_pflags_ana" )


::

    In [13]: a.hismask.abbr2code                                                                                                                                                                            
    Out[13]: 
    {'CK': 1,
     'SI': 2,
     'MI': 4,
     'AB': 8,
     'RE': 16,
     'SC': 32,
     'SD': 64,
     'SA': 128,
     'DR': 256,
     'SR': 512,
     'BR': 1024,
     'BT': 2048,
     'TO': 4096,
     'NA': 8192,
     'EX': 16384,
     'EC': 32768,
     '_L': 65536,
     '_Y': 131072,
     '_E': 262144,
     'PE': 524288,
     'GE': 1048576}

::

    In [24]: ecex = a.hismask.code("EC|EX")
    Out[24]: 49152

    f = a.hismask.code("CK|SI|MI|AB|RE|SD|BT|BR|EC|EX") 

    In [31]: a.hismask.label(f)                                                                                                                                                                             
    Out[31]: 'EC|EX|BT|BR|SD|RE|AB|MI|SI|CK'

    In [32]: a.hismask.label(f & ~ecex )                                                                                                                                                                    
    Out[32]: 'BT|BR|SD|RE|AB|MI|SI|CK'




::

    tds3gun.sh 1

    In [9]: ab.flg[:40]
    Out[9]:
    ab.flg
    .       pflags_ana  cfo:sum  1:g4live:tds3gun   -1:g4live:tds3gun        c2        ab        ba
    .                              11278     11278      7348.87/35 = 209.97  (pval:0.000 prob:1.000)
    0000                a      1653      1665    -12             0.04        0.993 +- 0.024        1.007 +- 0.025  [2 ] AB|SI
    0001              882      1200      1224    -24             0.24        0.980 +- 0.028        1.020 +- 0.029  [3 ] BT|SA|SI
    0002              8a2       680       766    -86             5.11        0.888 +- 0.034        1.126 +- 0.041  [4 ] BT|SA|SC|SI
    0003              842         0      1389   -1389          1389.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] BT|SD|SI
    0004               1a       586       742   -156            18.33        0.790 +- 0.033        1.266 +- 0.046  [3 ] RE|AB|SI
    0005               2a       601       540     61             3.26        1.113 +- 0.045        0.899 +- 0.039  [3 ] SC|AB|SI
    0006              862         0      1020   -1020          1020.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] BT|SD|SC|SI
    0007              892       422       522   -100            10.59        0.808 +- 0.039        1.237 +- 0.054  [4 ] BT|SA|RE|SI
    0008              8b2       384       478    -94            10.25        0.803 +- 0.041        1.245 +- 0.057  [5 ] BT|SA|SC|RE|SI
    0009               3a       431       421     10             0.12        1.024 +- 0.049        0.977 +- 0.048  [4 ] SC|RE|AB|SI
    0010             4842       797         0    797           797.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] EX|BT|SD|SI
    0011              852         0       662   -662           662.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] BT|SD|RE|SI
    0012             4862       591         0    591           591.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] EX|BT|SD|SC|SI
    0013              82a       419       170    249           105.26        2.465 +- 0.120        0.406 +- 0.031  [4 ] BT|SC|AB|SI
    0014              80a       386       196    190            62.03        1.969 +- 0.100        0.508 +- 0.036  [3 ] BT|AB|SI
    0015              872         0       569   -569           569.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] BT|SD|SC|RE|SI
    0016             8842       548         0    548           548.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] EC|BT|SD|SI
    0017              83a       282       102    180            84.38        2.765 +- 0.165        0.362 +- 0.036  [5 ] BT|SC|RE|AB|SI
    0018             8862       352         0    352           352.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] EC|BT|SD|SC|SI
    0019             4852       339         0    339           339.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] EX|BT|SD|RE|SI
    0020             4872       313         0    313           313.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] EX|BT|SD|SC|RE|SI
    0021              81a       202        93    109            40.27        2.172 +- 0.153        0.460 +- 0.048  [4 ] BT|RE|AB|SI
    0022                9       142       144     -2             0.01        0.986 +- 0.083        1.014 +- 0.085  [2 ] AB|CK
    0023             8872       217         0    217           217.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] EC|BT|SD|SC|RE|SI
    0024             8852       198         0    198           198.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] EC|BT|SD|RE|SI
    0025              832        78       110    -32             5.45        0.709 +- 0.080        1.410 +- 0.134  [4 ] BT|SC|RE|SI
    0026              c32        54        47      7             0.49        1.149 +- 0.156        0.870 +- 0.127  [5 ] BT|BR|SC|RE|SI
    0027              c22        39        45     -6             0.43        0.867 +- 0.139        1.154 +- 0.172  [4 ] BT|BR|SC|SI
    0028              ca2        31        28      3             0.15        1.107 +- 0.199        0.903 +- 0.171  [5 ] BT|BR|SA|SC|SI
    0029              c2a        30        18     12             3.00        1.667 +- 0.304        0.600 +- 0.141  [5 ] BT|BR|SC|AB|SI
    0030               19        26        21      5             0.53        1.238 +- 0.243        0.808 +- 0.176  [3 ] RE|AB|CK
    0031               32        15        26    -11             2.95        0.577 +- 0.149        1.733 +- 0.340  [3 ] SC|RE|SI
    0032              891        22        17      5             0.64        1.294 +- 0.276        0.773 +- 0.187  [4 ] BT|SA|RE|CK
    0033              aa2        20        19      1             0.03        1.053 +- 0.235        0.950 +- 0.218  [5 ] BT|SR|SA|SC|SI
    0034              c82        16        17     -1             0.03        0.941 +- 0.235        1.062 +- 0.258  [4 ] BT|BR|SA|SI
    0035              c3a        18        15      3             0.27        1.200 +- 0.283        0.833 +- 0.215  [6 ] BT|BR|SC|RE|AB|SI
    0036              cb2        18        12      6             0.00        1.500 +- 0.354        0.667 +- 0.192  [6 ] BT|BR|SA|SC|RE|SI
    0037              822         9        16     -7             0.00        0.562 +- 0.188        1.778 +- 0.444  [3 ] BT|SC|SI
    0038              c92        10        13     -3             0.00        0.769 +- 0.243        1.300 +- 0.361  [5 ] BT|BR|SA|RE|SI
    0039              871         0        21    -21             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] BT|SD|SC|RE|CK
    .                              11278     11278      7348.87/35 = 209.97  (pval:0.000 prob:1.000)




    Out[7]:
    ab.flg
    .       pflags_ana  cfo:sum  1:g4live:tds3gun   -1:g4live:tds3gun        c2        ab        ba
    .                              11278     11278      7348.87/35 = 209.97  (pval:0.000 prob:1.000)
    0000                a      1653      1665    -12             0.04        0.993 +- 0.024        1.007 +- 0.025  [2 ] AB|SI
    0001              882      1200      1224    -24             0.24        0.980 +- 0.028        1.020 +- 0.029  [3 ] BT|SA|SI
    0002              8a2       680       766    -86             5.11        0.888 +- 0.034        1.126 +- 0.041  [4 ] BT|SA|SC|SI
    0003              842         0      1389   -1389          1389.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] BT|SD|SI
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    0004               1a       586       742   -156            18.33        0.790 +- 0.033        1.266 +- 0.046  [3 ] RE|AB|SI
    0005               2a       601       540     61             3.26        1.113 +- 0.045        0.899 +- 0.039  [3 ] SC|AB|SI
    0006              862         0      1020   -1020          1020.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] BT|SD|SC|SI
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    0007              892       422       522   -100            10.59        0.808 +- 0.039        1.237 +- 0.054  [4 ] BT|SA|RE|SI
    0008              8b2       384       478    -94            10.25        0.803 +- 0.041        1.245 +- 0.057  [5 ] BT|SA|SC|RE|SI
    0009               3a       431       421     10             0.12        1.024 +- 0.049        0.977 +- 0.048  [4 ] SC|RE|AB|SI
    0010             4842       797         0    797           797.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] EX|BT|SD|SI
    0011              852         0       662   -662           662.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] BT|SD|RE|SI
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    0012             4862       591         0    591           591.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] EX|BT|SD|SC|SI
    0013              82a       419       170    249           105.26        2.465 +- 0.120        0.406 +- 0.031  [4 ] BT|SC|AB|SI
    0014              80a       386       196    190            62.03        1.969 +- 0.100        0.508 +- 0.036  [3 ] BT|AB|SI
    0015              872         0       569   -569           569.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] BT|SD|SC|RE|SI
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    0016             8842       548         0    548           548.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] EC|BT|SD|SI
    0017              83a       282       102    180            84.38        2.765 +- 0.165        0.362 +- 0.036  [5 ] BT|SC|RE|AB|SI
    0018             8862       352         0    352           352.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] EC|BT|SD|SC|SI
    0019             4852       339         0    339           339.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] EX|BT|SD|RE|SI
    .                              11278     11278      7348.87/35 = 209.97  (pval:0.000 prob:1.000)




evt.py::

     581         ecex = self.hismask.code("EC|EX")
     582         all_pflags_ana = self.make_pflags_ana( self.pflags & ~ecex , "all_pflags_ana" )  # SCRUB "EC|EX" **TEMPORARILY**
     583         
     584         self.all_pflags_ana = all_pflags_ana
     585         self.pflags_ana = all_pflags_ana
     586         



::

    In [5]: a.all_pflags_ana.table[:20]                                                                                                                                                                     
    Out[5]: 
    all_pflags_ana
    .                     cfo:-  1:g4live:tds3gun 
    .                              11278         1.00 
    0000                a        0.147        1653        [2 ] AB|SI
    0001              882        0.106        1200        [3 ] BT|SA|SI
    0002             4842        0.071         797        [4 ] EX|BT|SD|SI
    0003              8a2        0.060         680        [4 ] BT|SA|SC|SI
    0004               2a        0.053         601        [3 ] SC|AB|SI
    0005             4862        0.052         591        [5 ] EX|BT|SD|SC|SI
    0006               1a        0.052         586        [3 ] RE|AB|SI
    0007             8842        0.049         548        [4 ] EC|BT|SD|SI
    0008               3a        0.038         431        [4 ] SC|RE|AB|SI
    0009              892        0.037         422        [4 ] BT|SA|RE|SI
    0010              82a        0.037         419        [4 ] BT|SC|AB|SI
    0011              80a        0.034         386        [3 ] BT|AB|SI
    0012              8b2        0.034         384        [5 ] BT|SA|SC|RE|SI
    0013             8862        0.031         352        [5 ] EC|BT|SD|SC|SI
    0014             4852        0.030         339        [5 ] EX|BT|SD|RE|SI
    0015             4872        0.028         313        [6 ] EX|BT|SD|SC|RE|SI
    0016              83a        0.025         282        [5 ] BT|SC|RE|AB|SI
    0017             8872        0.019         217        [6 ] EC|BT|SD|SC|RE|SI
    0018              81a        0.018         202        [4 ] BT|RE|AB|SI
    0019             8852        0.018         198        [5 ] EC|BT|SD|RE|SI
    .                              11278         1.00 

    In [6]: a.all_pflags_ana2.table[:20]                                                                                                                                                                    
    Out[6]: 
    all_pflags_ana
    .                     cfo:-  1:g4live:tds3gun 
    .                              11278         1.00 
    0000                a        0.147        1653        [2 ] AB|SI
    0001              842        0.119        1345        [3 ] BT|SD|SI
    0002              882        0.106        1200        [3 ] BT|SA|SI
    0003              862        0.084         943        [4 ] BT|SD|SC|SI
    0004              8a2        0.060         680        [4 ] BT|SA|SC|SI
    0005               2a        0.053         601        [3 ] SC|AB|SI
    0006               1a        0.052         586        [3 ] RE|AB|SI
    0007              852        0.048         537        [4 ] BT|SD|RE|SI
    0008              872        0.047         530        [5 ] BT|SD|SC|RE|SI
    0009               3a        0.038         431        [4 ] SC|RE|AB|SI
    0010              892        0.037         422        [4 ] BT|SA|RE|SI
    0011              82a        0.037         419        [4 ] BT|SC|AB|SI
    0012              80a        0.034         386        [3 ] BT|AB|SI
    0013              8b2        0.034         384        [5 ] BT|SA|SC|RE|SI
    0014              83a        0.025         282        [5 ] BT|SC|RE|AB|SI
    0015              81a        0.018         202        [4 ] BT|RE|AB|SI
    0016                9        0.013         142        [2 ] AB|CK
    0017              832        0.007          78        [4 ] BT|SC|RE|SI
    0018              c32        0.005          54        [5 ] BT|BR|SC|RE|SI
    0019              c22        0.003          39        [4 ] BT|BR|SC|SI
    .                              11278         1.00 

    In [7]:                                                                                                                                                                                                 



