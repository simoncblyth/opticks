tconcentric pflags mismatch from_truncation handling
======================================================

Status
--------

Dumping indicates a mismatch in handling of truncation following the 
move from live to canned recording.

Suspect issue is from RecordStepPoint being developed for live running where
records are written until truncation is hit and the method returns done = true to 
indicate truncation.  

That works OK in canned mode for a single G4 track, but with rejoining of reemtracks
the CRecorder::CannedWriteSteps gets called for each of the rejoin attempts and
then it is incorrect to plough ahead with the RecordStepPoint when already in truncation.

So need to know the "done" status before, not after in canned running.


::

     828 #ifdef USE_CUSTOM_BOUNDARY
     829 bool CRecorder::RecordStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, DsG4OpBoundaryProcessStatus boundary_status, const char* label)
     830 #else
     831 bool CRecorder::RecordStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, G4OpBoundaryProcessStatus boundary_status, const char* label)
     832 #endif
        



Issue
-------

pflags obtained from seqhis and direct pflags obtained from photon buffer 
are mismatched for 2 out of 1M photons : probably a truncation difference


::

    simon:opticks blyth$ tconcentric-i 
    ...
    /Users/blyth/opticks/ana/tconcentric.py --tag 1 --det concentric --src torch
    [2016-11-22 11:46:40,190] p97491 {/Users/blyth/opticks/ana/evt.py:431} INFO - pflags2(=seq2msk(seqhis)) and pflags  MISMATCH (msk_mismatch)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AB(1,torch,concentric)  None 0 
    A concentric/torch/  1 :  20161122-1111 maxbounce:15 maxrec:16 maxrng:3000000 /tmp/blyth/opticks/evt/concentric/torch/1/fdom.npy 
    B concentric/torch/ -1 :  20161122-1111 maxbounce:15 maxrec:16 maxrng:3000000 /tmp/blyth/opticks/evt/concentric/torch/-1/fdom.npy 
    .                seqhis_ana      noname       noname           c2           ab           ba 
    .                               1000000      1000000       371.33/355 =  1.05  (pval:0.265 prob:0.735)  
       0               8ccccd        669843       670001             0.02        1.000 +- 0.001        1.000 +- 0.001  [6 ] TO BT BT BT BT SA
       1                   4d         83950        84149             0.24        0.998 +- 0.003        1.002 +- 0.003  [2 ] TO AB
       2              8cccc6d         45490        44770             5.74        1.016 +- 0.005        0.984 +- 0.005  [7 ] TO SC BT BT BT BT SA
       3               4ccccd         28955        28718             0.97        1.008 +- 0.006        0.992 +- 0.006  [6 ] TO BT BT BT BT AB
       4                 4ccd         23187        23170             0.01        1.001 +- 0.007        0.999 +- 0.007  [4 ] TO BT BT AB
       5              8cccc5d         20238        20140             0.24        1.005 +- 0.007        0.995 +- 0.007  [7 ] TO RE BT BT BT BT SA
       6              8cc6ccd         10214        10357             0.99        0.986 +- 0.010        1.014 +- 0.010  [7 ] TO BT BT SC BT BT SA
       7              86ccccd         10176        10318             0.98        0.986 +- 0.010        1.014 +- 0.010  [7 ] TO BT BT BT BT SC SA
       8              89ccccd          7540         7710             1.90        0.978 +- 0.011        1.023 +- 0.012  [7 ] TO BT BT BT BT DR SA
       9             8cccc55d          5976         5934             0.15        1.007 +- 0.013        0.993 +- 0.013  [8 ] TO RE RE BT BT BT BT SA
      10                  45d          5779         5766             0.01        1.002 +- 0.013        0.998 +- 0.013  [3 ] TO RE AB
      11      8cccccccc9ccccd          5339         5269             0.46        1.013 +- 0.014        0.987 +- 0.014  [15] TO BT BT BT BT DR BT BT BT BT BT BT BT BT SA
      12              8cc5ccd          5111         4940             2.91        1.035 +- 0.014        0.967 +- 0.014  [7 ] TO BT BT RE BT BT SA
      13                  46d          4797         4886             0.82        0.982 +- 0.014        1.019 +- 0.015  [3 ] TO SC AB
      14          8cccc9ccccd          4494         4469             0.07        1.006 +- 0.015        0.994 +- 0.015  [11] TO BT BT BT BT DR BT BT BT BT SA
      15          8cccccc6ccd          3317         3302             0.03        1.005 +- 0.017        0.995 +- 0.017  [11] TO BT BT SC BT BT BT BT BT BT SA
      16             8cccc66d          2670         2675             0.00        0.998 +- 0.019        1.002 +- 0.019  [8 ] TO SC SC BT BT BT BT SA
      17              49ccccd          2432         2383             0.50        1.021 +- 0.021        0.980 +- 0.020  [7 ] TO BT BT BT BT DR AB
      18              4cccc6d          2043         1991             0.67        1.026 +- 0.023        0.975 +- 0.022  [7 ] TO SC BT BT BT BT AB
      19                4cc6d          1755         1826             1.41        0.961 +- 0.023        1.040 +- 0.024  [5 ] TO SC BT BT AB
    .                               1000000      1000000       371.33/355 =  1.05  (pval:0.265 prob:0.735)  
    .                pflags_ana  1:concentric   -1:concentric           c2           ab           ba 
    .                               1000000      1000000        42.43/42 =  1.01  (pval:0.453 prob:0.547)  
       0                 1880        669843       670001             0.02        1.000 +- 0.001        1.000 +- 0.001  [3 ] TO|BT|SA
       1                 1008         83950        84149             0.24        0.998 +- 0.003        1.002 +- 0.003  [2 ] TO|AB
       2                 18a0         79906        79434             1.40        1.006 +- 0.004        0.994 +- 0.004  [4 ] TO|BT|SA|SC
       3                 1808         54172        53846             0.98        1.006 +- 0.004        0.994 +- 0.004  [3 ] TO|BT|AB
       4                 1890         38515        38398             0.18        1.003 +- 0.005        0.997 +- 0.005  [4 ] TO|BT|SA|RE
       5                 1980         17710        17792             0.19        0.995 +- 0.007        1.005 +- 0.008  [4 ] TO|BT|DR|SA
       6                 1828          8788         9082             4.84        0.968 +- 0.010        1.033 +- 0.011  [4 ] TO|BT|SC|AB
       7                 1018          8200         8120             0.39        1.010 +- 0.011        0.990 +- 0.011  [3 ] TO|RE|AB
       8                 18b0          7902         7957             0.19        0.993 +- 0.011        1.007 +- 0.011  [5 ] TO|BT|SA|SC|RE
       9                 1818          6027         6157             1.39        0.979 +- 0.013        1.022 +- 0.013  [4 ] TO|BT|RE|AB
      10                 1908          5531         5410             1.34        1.022 +- 0.014        0.978 +- 0.013  [4 ] TO|BT|DR|AB
      11                 1028          5089         5208             1.38        0.977 +- 0.014        1.023 +- 0.014  [3 ] TO|SC|AB
      12                 19a0          4931         4972             0.17        0.992 +- 0.014        1.008 +- 0.014  [5 ] TO|BT|DR|SA|SC
      13                 1990          1482         1556             1.80        0.952 +- 0.025        1.050 +- 0.027  [5 ] TO|BT|DR|SA|RE
      14                 1838          1541         1550             0.03        0.994 +- 0.025        1.006 +- 0.026  [5 ] TO|BT|SC|RE|AB
      15                 1928          1056         1086             0.42        0.972 +- 0.030        1.028 +- 0.031  [5 ] TO|BT|DR|SC|AB
      16                 1920           789          744             1.32        1.060 +- 0.038        0.943 +- 0.035  [4 ] TO|BT|DR|SC
      17                 1038           769          785             0.16        0.980 +- 0.035        1.021 +- 0.036  [4 ] TO|SC|RE|AB
      18                 1918           628          599             0.69        1.048 +- 0.042        0.954 +- 0.039  [5 ] TO|BT|DR|RE|AB
      19                 1910           493          433             3.89        1.139 +- 0.051        0.878 +- 0.042  [4 ] TO|BT|DR|RE
    .                               1000000      1000000        42.43/42 =  1.01  (pval:0.453 prob:0.547)  



Selecting mismatch using PFLAGS_DEBUG
------------------------------------------

The mask has a DR that does not appear in seqhis::

    In [2]: b = ab.b    ## or run evt.py 

    In [3]: b.sel = "PFLAGS_DEBUG"  ; print b.his, b.mat, b.flg   
    .                            -1:concentric 
    .                                     2         1.00 
       0     c5cccccccc6ccccd        1.000              2         [16] TO BT BT BT BT SC BT BT BT BT BT BT BT BT RE BT
    .                                     2         1.00  .                            -1:concentric 
    .                                     2         1.00 
       0     3243231323443231        0.500              1         [16] Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO LS Ac
       1     3143231323443231        0.500              1         [16] Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO Gd Ac
    .                                     2         1.00  .                            -1:concentric 
    .                                     2         1.00 
       0                 1930        1.000              2         [5 ] TO|BT|DR|SC|RE
    .                                     2         1.00 

    In [4]: b.psel_dindex()
    Out[4]: '--dindex=430603,521493'



    In [9]: b.pflags, b.pflags2
    Out[9]: 
    A([6152, 6272, 6272, ..., 6272, 6272, 6272], dtype=uint32),
    A([6152, 6272, 6272, ..., 6272, 6272, 6272], dtype=uint64))

    In [8]: b.pflags == b.pflags2
    Out[8]: 
    A()sliced
    A([ True,  True,  True, ...,  True,  True,  True], dtype=bool)

    In [9]: np.count_nonzero(b.pflags == b.pflags2)
    Out[9]: 999998

    In [10]: np.count_nonzero(b.pflags != b.pflags2)    ## 2 in 1M issue
    Out[10]: 2



Look at seqhis and seqmat within that mask, note that all are truncated and rejoined.
Actually they must be truncated as no absorption in the mask AB/SA/SD, and as RE they must be rejoined.

::

    In [6]: b.selflg = "TO|BT|DR|SC|RE" ; print b.his[:10] ; print b.mat[:10] ; print b.flg
    .                            -1:concentric 
    .                                   424         1.00 
       0     cccc65cccc9ccccd        0.050             21         [16] TO BT BT BT BT DR BT BT BT BT RE SC BT BT BT BT
       1     cccc6cccc9cccc5d        0.040             17         [16] TO RE BT BT BT BT DR BT BT BT BT SC BT BT BT BT
       2     cccc56cccc9ccccd        0.031             13         [16] TO BT BT BT BT DR BT BT BT BT SC RE BT BT BT BT
       3     cccc5cccc9cccc6d        0.031             13         [16] TO SC BT BT BT BT DR BT BT BT BT RE BT BT BT BT
       4     cccccccc9cccc65d        0.031             13         [16] TO RE SC BT BT BT BT DR BT BT BT BT BT BT BT BT
       5     6cccc5cccc9ccccd        0.021              9         [16] TO BT BT BT BT DR BT BT BT BT RE BT BT BT BT SC
       6     cccccccc9cccc56d        0.021              9         [16] TO SC RE BT BT BT BT DR BT BT BT BT BT BT BT BT
       7     cc5cccccc9cccc6d        0.019              8         [16] TO SC BT BT BT BT DR BT BT BT BT BT BT RE BT BT
       8     cccc5cccc96ccccd        0.019              8         [16] TO BT BT BT BT SC DR BT BT BT BT RE BT BT BT BT
       9     cccc5cc6cc9ccccd        0.019              8         [16] TO BT BT BT BT DR BT BT SC BT BT RE BT BT BT BT
    .                                   424         1.00 
    .                            -1:concentric 
    .                                   424         1.00 
       0     4323111323443231        0.080             34         [16] Gd Ac LS Ac MO MO Ac LS Ac Gd Gd Gd Ac LS Ac MO
       1     4323113234432311        0.068             29         [16] Gd Gd Ac LS Ac MO MO Ac LS Ac Gd Gd Ac LS Ac MO
       2     4323132344323111        0.052             22         [16] Gd Gd Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO
       3     4432311323443231        0.028             12         [16] Gd Ac LS Ac MO MO Ac LS Ac Gd Gd Ac LS Ac MO MO
       4     4322313234432311        0.026             11         [16] Gd Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS LS Ac MO
       5     4323113223443231        0.026             11         [16] Gd Ac LS Ac MO MO Ac LS LS Ac Gd Gd Ac LS Ac MO
       6     4323113234443231        0.026             11         [16] Gd Ac LS Ac MO MO MO Ac LS Ac Gd Gd Ac LS Ac MO
       7     4322311323443231        0.024             10         [16] Gd Ac LS Ac MO MO Ac LS Ac Gd Gd Ac LS LS Ac MO
       8     4323113234432231        0.019              8         [16] Gd Ac LS LS Ac MO MO Ac LS Ac Gd Gd Ac LS Ac MO
       9     3231111323443231        0.019              8         [16] Gd Ac LS Ac MO MO Ac LS Ac Gd Gd Gd Gd Ac LS Ac
    .                                   424         1.00 
    .                            -1:concentric 
    .                                   424         1.00 
       0                 1930        1.000            424         [5 ] TO|BT|DR|SC|RE
    .                                   424         1.00 





Dump first mismatch photon
----------------------------



::

    tconcentric-;tconcentric-tt --dindex=430603

Note that there are multiple reemtracks but all past truncation. 
The seqhis and mask starts out correct from the first G4 track.. except there is no RE of course.

::

    2016-11-23 12:45:18.488 INFO  [1772513] [CTrackingAction::dump@186] CTrackingAction::setPhotonId --dindex  record_id 430603 event_id 43 track_id 603 photon_id 603 parent_id -1 primary_id -2 reemtrack 0
    2016-11-23 12:45:18.489 INFO  [1772513] [CRecorder::dump@1218] CRecorder::posttrack
    2016-11-23 12:45:18.489 INFO  [1772513] [CRecorder::dump_brief@1230] CRecorder::dump_brief m_record_id   430603 m_badflag     0 --dindex 
    2016-11-23 12:45:18.489 INFO  [1772513] [CRecorder::dump_brief@1238]  seqhis c9cccccccc6ccccd    TO BT BT BT BT SC BT BT BT BT BT BT BT BT DR BT 
    2016-11-23 12:45:18.489 INFO  [1772513] [CRecorder::dump_brief@1243]  mskhis             1920    SC|DR|BT|TO
    2016-11-23 12:45:18.489 INFO  [1772513] [CRecorder::dump_brief@1248]  seqmat 3443231323443231    Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO MO Ac 
    2016-11-23 12:45:18.490 INFO  [1772513] [CRecorder::dump_sequence@1257] CRecorder::dump_sequence
       0                d TO                                              
       1               cd TO BT                                           
       2              ccd TO BT BT                                        
       3             cccd TO BT BT BT                                     
       4            ccccd TO BT BT BT BT                                  
       5           6ccccd TO BT BT BT BT SC                               
       6          c6ccccd TO BT BT BT BT SC BT                            
       7         cc6ccccd TO BT BT BT BT SC BT BT                         
       8        ccc6ccccd TO BT BT BT BT SC BT BT BT                      
       9       cccc6ccccd TO BT BT BT BT SC BT BT BT BT                   
      10      ccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT                
      11     cccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT BT             
      12    ccccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT BT BT          
      13   cccccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT BT BT BT       
      14  9cccccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT BT BT BT DR    
      15 c9cccccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT BT BT BT DR BT 
       0             1000 TO
       1             1800 BT|TO
       2             1800 BT|TO
       3             1800 BT|TO
       4             1800 BT|TO
       5             1820 SC|BT|TO
       6             1820 SC|BT|TO
       7             1820 SC|BT|TO
       8             1820 SC|BT|TO
       9             1820 SC|BT|TO
      10             1820 SC|BT|TO
      11             1820 SC|BT|TO
      12             1820 SC|BT|TO
      13             1820 SC|BT|TO
      14             1920 SC|DR|BT|TO
      15             1920 SC|DR|BT|TO
       0                1 Gd - - - - - - - - - - - - - - - 
       1               31 Gd Ac - - - - - - - - - - - - - - 
       2              231 Gd Ac LS - - - - - - - - - - - - - 
       3             3231 Gd Ac LS Ac - - - - - - - - - - - - 
       4            43231 Gd Ac LS Ac MO - - - - - - - - - - - 
       5           443231 Gd Ac LS Ac MO MO - - - - - - - - - - 
       6          3443231 Gd Ac LS Ac MO MO Ac - - - - - - - - - 
       7         23443231 Gd Ac LS Ac MO MO Ac LS - - - - - - - - 
       8        323443231 Gd Ac LS Ac MO MO Ac LS Ac - - - - - - - 
       9       1323443231 Gd Ac LS Ac MO MO Ac LS Ac Gd - - - - - - 
      10      31323443231 Gd Ac LS Ac MO MO Ac LS Ac Gd Ac - - - - - 
      11     231323443231 Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS - - - - 
      12    3231323443231 Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac - - - 
      13   43231323443231 Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO - - 
      14  443231323443231 Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO MO - 
      15 3443231323443231 Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO MO Ac 


Despite already being past truncation from the first track::

    (12)  BT/BT     FrT                                                     
    [  12](Stp ;opticalphoton stepNum   20(tk ;opticalphoton tid 604 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ -1017.584-2339.954 666.007]  )
      pre               sphere_phys         Acrylic  Transportation        GeomBoundary pos[   -101.728 -3515.686 -1894.632]  dir[   -0.741  -0.591  -0.318]  pol[    0.541  -0.807   0.237]  ns 53.164 nm 430.000 mm/ns 192.780
     post               sphere_phys      MineralOil  Transportation        GeomBoundary pos[   -112.461 -3524.239 -1899.241]  dir[   -0.755  -0.577  -0.311]  pol[    0.529  -0.817   0.231]  ns 53.239 nm 430.000 mm/ns 197.134
     )
    (13)  BT/DR     LaR                                            MAT_SWAP 
    [  13](Stp ;opticalphoton stepNum   20(tk ;opticalphoton tid 604 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ -1017.584-2339.954 666.007]  )
      pre               sphere_phys      MineralOil  Transportation        GeomBoundary pos[   -112.461 -3524.239 -1899.241]  dir[   -0.755  -0.577  -0.311]  pol[    0.529  -0.817   0.231]  ns 53.239 nm 430.000 mm/ns 197.134
     post               sphere_phys         Acrylic  Transportation        GeomBoundary pos[  -1111.371 -4286.912 -2310.252]  dir[    0.023   0.541   0.841]  pol[   -0.644   0.651  -0.401]  ns 59.946 nm 430.000 mm/ns 197.134
     )
    (14)  DR/NA     STS                                           POST_SKIP 
    [  14](Stp ;opticalphoton stepNum   20(tk ;opticalphoton tid 604 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ -1017.584-2339.954 666.007]  )
      pre               sphere_phys         Acrylic  Transportation        GeomBoundary pos[  -1111.371 -4286.912 -2310.252]  dir[    0.023   0.541   0.841]  pol[   -0.644   0.651  -0.401]  ns 59.946 nm 430.000 mm/ns 197.134
     post               sphere_phys      MineralOil  Transportation        GeomBoundary pos[  -1111.371 -4286.912 -2310.252]  dir[    0.023   0.541   0.841]  pol[   -0.644   0.651  -0.401]  ns 59.946 nm 430.000 mm/ns 197.134
     )
    (15)  NA/BT     FrT                     RECORD_TRUNCATE BOUNCE_TRUNCATE 
    [  15](Stp ;opticalphoton stepNum   20(tk ;opticalphoton tid 604 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ -1017.584-2339.954 666.007]  )
      pre               sphere_phys      MineralOil  Transportation        GeomBoundary pos[  -1111.371 -4286.912 -2310.252]  dir[    0.023   0.541   0.841]  pol[   -0.644   0.651  -0.401]  ns 59.946 nm 430.000 mm/ns 197.134
     post               sphere_phys         Acrylic  Transportation        GeomBoundary pos[  -1083.436 -3632.430 -1292.920]  dir[    0.030   0.553   0.832]  pol[   -0.644   0.648  -0.408]  ns 66.084 nm 430.000 mm/ns 192.780
     )
    (16)  BT/BT     FrT                       RECORD_TRUNCATE HARD_TRUNCATE 
    [  16](Stp ;opticalphoton stepNum   20(tk ;opticalphoton tid 604 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ -1017.584-2339.954 666.007]  )
      pre               sphere_phys         Acrylic  Transportation        GeomBoundary pos[  -1083.436 -3632.430 -1292.920]  dir[    0.030   0.553   0.832]  pol[   -0.644   0.648  -0.408]  ns 66.084 nm 430.000 mm/ns 192.780
     post               sphere_phys uidScintillator  Transportation        GeomBoundary pos[  -1083.056 -3625.318 -1282.220]  dir[    0.028   0.550   0.834]  pol[   -0.644   0.648  -0.406]  ns 66.151 nm 430.000 mm/ns 194.519
     )
    (17)  BT/BT     FrT                       RECORD_TRUNCATE HARD_TRUNCATE 
    [  17](Stp ;opticalphoton stepNum   20(tk ;opticalphoton tid 604 pid 0 nm    430 mm  ori[    0.000   0.000   0.000]  pos[ -1017.584-2339.954 666.007]  )
      pre               sphere_phys uidScintillator  Transportation        GeomBoundary pos[  -1083.056 -3625.318 -1282.220]  dir[    0.028   0.550   0.834]  pol[   -0.644   0.648  -0.406]  ns 66.151 nm 430.000 mm/ns 194.519
     post               sphere_phys         Acrylic  Transportation        GeomBoundary pos[  -1041.946 -2817.970   -58.457]  dir[    0.031   0.556   0.831]  pol[   -0.644   0.647  -0.409]  ns 73.691 nm 430.000 mm/ns 192.780
     )


First reemtrack doesnt change anything as already in truncation (that is the policy ?)::

    2016-11-23 12:45:18.494 INFO  [1772513] [CTrackingAction::dump@186] CTrackingAction::setPhotonId --dindex  record_id 430603 event_id 43 track_id 10893 photon_id 603 parent_id 603 primary_id -2 reemtrack 1
    2016-11-23 12:45:18.494 INFO  [1772513] [CRecorder::dump@1218] CRecorder::posttrack
    2016-11-23 12:45:18.494 INFO  [1772513] [CRecorder::dump_brief@1230] CRecorder::dump_brief m_record_id   430603 m_badflag     0 --dindex 
    2016-11-23 12:45:18.494 INFO  [1772513] [CRecorder::dump_brief@1238]  seqhis c9cccccccc6ccccd    TO BT BT BT BT SC BT BT BT BT BT BT BT BT DR BT 
    2016-11-23 12:45:18.494 INFO  [1772513] [CRecorder::dump_brief@1243]  mskhis             1920    SC|DR|BT|TO
    2016-11-23 12:45:18.494 INFO  [1772513] [CRecorder::dump_brief@1248]  seqmat 3443231323443231    Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO MO Ac 
    2016-11-23 12:45:18.494 INFO  [1772513] [CRecorder::dump_sequence@1257] CRecorder::dump_sequence
       0                d TO                                              
       1               cd TO BT                                           
       2              ccd TO BT BT                                        
       3             cccd TO BT BT BT                                     
       4            ccccd TO BT BT BT BT                                  
       5           6ccccd TO BT BT BT BT SC                               
       6          c6ccccd TO BT BT BT BT SC BT                            
       7         cc6ccccd TO BT BT BT BT SC BT BT                         
       8        ccc6ccccd TO BT BT BT BT SC BT BT BT                      
       9       cccc6ccccd TO BT BT BT BT SC BT BT BT BT                   
      10      ccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT                
      11     cccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT BT             
      12    ccccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT BT BT          
      13   cccccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT BT BT BT       
      14  9cccccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT BT BT BT DR    
      15 c9cccccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT BT BT BT DR BT 
       0             1000 TO
       1             1800 BT|TO
       2             1800 BT|TO
       3             1800 BT|TO
       4             1800 BT|TO
       5             1820 SC|BT|TO
       6             1820 SC|BT|TO
       7             1820 SC|BT|TO
       8             1820 SC|BT|TO
       9             1820 SC|BT|TO
      10             1820 SC|BT|TO
      11             1820 SC|BT|TO
      12             1820 SC|BT|TO
      13             1820 SC|BT|TO
      14             1920 SC|DR|BT|TO
      15             1920 SC|DR|BT|TO


Second reemtrack somehow manages to replace the DR BT with RE BT::

    2016-11-23 12:45:18.495 INFO  [1772513] [CTrackingAction::dump@186] CTrackingAction::setPhotonId --dindex  record_id 430603 event_id 43 track_id 10894 photon_id 603 parent_id 10893 primary_id 603 reemtrack 1
    2016-11-23 12:45:18.495 INFO  [1772513] [CRecorder::dump@1218] CRecorder::posttrack
    2016-11-23 12:45:18.495 INFO  [1772513] [CRecorder::dump_brief@1230] CRecorder::dump_brief m_record_id   430603 m_badflag     0 --dindex 
    2016-11-23 12:45:18.495 INFO  [1772513] [CRecorder::dump_brief@1238]  seqhis c5cccccccc6ccccd    TO BT BT BT BT SC BT BT BT BT BT BT BT BT RE BT 
    2016-11-23 12:45:18.495 INFO  [1772513] [CRecorder::dump_brief@1243]  mskhis             1930    RE|SC|DR|BT|TO
    2016-11-23 12:45:18.496 INFO  [1772513] [CRecorder::dump_brief@1248]  seqmat 3143231323443231    Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO Gd Ac 
    2016-11-23 12:45:18.496 INFO  [1772513] [CRecorder::dump_sequence@1257] CRecorder::dump_sequence
       0                d TO                                              
       1               cd TO BT                                           
       2              ccd TO BT BT                                        
       3             cccd TO BT BT BT                                     
       4            ccccd TO BT BT BT BT                                  
       5           6ccccd TO BT BT BT BT SC                               
       6          c6ccccd TO BT BT BT BT SC BT                            
       7         cc6ccccd TO BT BT BT BT SC BT BT                         
       8        ccc6ccccd TO BT BT BT BT SC BT BT BT                      
       9       cccc6ccccd TO BT BT BT BT SC BT BT BT BT                   
      10      ccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT                
      11     cccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT BT             
      12    ccccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT BT BT          
      13   cccccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT BT BT BT       
      14  9cccccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT BT BT BT DR    
      15 c9cccccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT BT BT BT DR BT 
      16 c5cccccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT BT BT BT RE BT 
       0             1000 TO
       1             1800 BT|TO
       2             1800 BT|TO
       3             1800 BT|TO
       4             1800 BT|TO
       5             1820 SC|BT|TO
       6             1820 SC|BT|TO
       7             1820 SC|BT|TO
       8             1820 SC|BT|TO
       9             1820 SC|BT|TO
      10             1820 SC|BT|TO
      11             1820 SC|BT|TO
      12             1820 SC|BT|TO
      13             1820 SC|BT|TO
      14             1920 SC|DR|BT|TO
      15             1920 SC|DR|BT|TO
      16             1930 RE|SC|DR|BT|TO



Hmm vague recollection of some special casing to match rejoined flags..



Dump 2nd mismatch
------------------

Same story:

* primary track already pushes to truncation
* 1st reemtrack fails to change this but 2nd reemtrack manages to chane ending "DR BT" to "RE BT"

::

    tconcentric-tt --dindex=521493


    2016-11-23 13:05:07.228 INFO  [1776602] [CTrackingAction::dump@186] CTrackingAction::setPhotonId --dindex  record_id 521493 event_id 52 track_id 1493 photon_id 1493 parent_id -1 primary_id -2 reemtrack 0
    2016-11-23 13:05:07.228 INFO  [1776602] [CRecorder::dump@1218] CRecorder::posttrack
    2016-11-23 13:05:07.228 INFO  [1776602] [CRecorder::dump_brief@1230] CRecorder::dump_brief m_record_id   521493 m_badflag     0 --dindex 
    2016-11-23 13:05:07.228 INFO  [1776602] [CRecorder::dump_brief@1238]  seqhis c9cccccccc6ccccd    TO BT BT BT BT SC BT BT BT BT BT BT BT BT DR BT 
    2016-11-23 13:05:07.228 INFO  [1776602] [CRecorder::dump_brief@1243]  mskhis             1920    SC|DR|BT|TO
    2016-11-23 13:05:07.228 INFO  [1776602] [CRecorder::dump_brief@1248]  seqmat 3443231323443231    Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO MO Ac 
    2016-11-23 13:05:07.228 INFO  [1776602] [CRecorder::dump_sequence@1257] CRecorder::dump_sequence
       0                d TO                                              
       1               cd TO BT                                           
       2              ccd TO BT BT                                        
       3             cccd TO BT BT BT                                     
       4            ccccd TO BT BT BT BT                                  
       5           6ccccd TO BT BT BT BT SC                               
       6          c6ccccd TO BT BT BT BT SC BT                            
       7         cc6ccccd TO BT BT BT BT SC BT BT                         
       8        ccc6ccccd TO BT BT BT BT SC BT BT BT                      
       9       cccc6ccccd TO BT BT BT BT SC BT BT BT BT                   
      10      ccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT                
      11     cccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT BT             
      12    ccccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT BT BT          
      13   cccccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT BT BT BT       
      14  9cccccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT BT BT BT DR    
      15 c9cccccccc6ccccd TO BT BT BT BT SC BT BT BT BT BT BT BT BT DR BT 
       0             1000 TO
       1             1800 BT|TO
       2             1800 BT|TO
       3             1800 BT|TO
       4             1800 BT|TO
       5             1820 SC|BT|TO
       6             1820 SC|BT|TO
       7             1820 SC|BT|TO
       8             1820 SC|BT|TO
       9             1820 SC|BT|TO
      10             1820 SC|BT|TO
      11             1820 SC|BT|TO
      12             1820 SC|BT|TO
      13             1820 SC|BT|TO
      14             1920 SC|DR|BT|TO
      15             1920 SC|DR|BT|TO
       0                1 Gd - - - - - - - - - - - - - - - 
       1               31 Gd Ac - - - - - - - - - - - - - - 
       2              231 Gd Ac LS - - - - - - - - - - - - - 
       3             3231 Gd Ac LS Ac - - - - - - - - - - - - 
       4            43231 Gd Ac LS Ac MO - - - - - - - - - - - 
       5           443231 Gd Ac LS Ac MO MO - - - - - - - - - - 
       6          3443231 Gd Ac LS Ac MO MO Ac - - - - - - - - - 
       7         23443231 Gd Ac LS Ac MO MO Ac LS - - - - - - - - 
       8        323443231 Gd Ac LS Ac MO MO Ac LS Ac - - - - - - - 
       9       1323443231 Gd Ac LS Ac MO MO Ac LS Ac Gd - - - - - - 
      10      31323443231 Gd Ac LS Ac MO MO Ac LS Ac Gd Ac - - - - - 
      11     231323443231 Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS - - - - 
      12    3231323443231 Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac - - - 
      13   43231323443231 Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO - - 
      14  443231323443231 Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO MO - 
      15 3443231323443231 Gd Ac LS Ac MO MO Ac LS Ac Gd Ac LS Ac MO MO Ac 


Truncation handling
----------------------

::

     815 #ifdef USE_CUSTOM_BOUNDARY
     816 bool CRecorder::RecordStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, DsG4OpBoundaryProcessStatus boundary_status, const char* label)
     817 #else
     818 bool CRecorder::RecordStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, G4OpBoundaryProcessStatus boundary_status, const char* label)
     819 #endif
     820 {
     821     // NB this is used by both the live and non-live "canned" modes of recording 
     822     //
     823     // Formerly at truncation, rerunning this overwrote "the top slot" 
     824     // of seqhis,seqmat bitfields (which are persisted in photon buffer)
     825     // and the record buffer. 
     826     // As that is different from Opticks behaviour for the record buffer
     827     // where truncation is truncation, a HARD_TRUNCATION has been adopted.
     ...
     833     m_record_truncate = slot == m_steps_per_photon - 1 ;    // hmm not exactly truncate, just top slot 
     834     if(m_record_truncate) m_step_action |= RECORD_TRUNCATE ;
     ...
     845     unsigned long long shift = slot*4ull ;     // 4-bits of shift for each slot 
     846     unsigned long long msk = 0xFull << shift ;
     847     unsigned long long his = BBit::ffs(flag) & 0xFull ;
     848     unsigned long long mat = material < 0xFull ? material : 0xFull ;
     849 
     850     unsigned long long prior_mat = ( m_seqmat & msk ) >> shift ;
     851     unsigned long long prior_his = ( m_seqhis & msk ) >> shift ;
     852     unsigned long long prior_flag = 0x1 << (prior_his - 1) ;
     853 
     854     if(m_record_truncate && prior_his != 0 && prior_mat != 0 )  // try to overwrite top slot 
     855     {
     856         m_topslot_rewrite += 1 ;
     857         LOG(info)
     858                   << ( m_topslot_rewrite > 1 ? HARD_TRUNCATE_ : TOPSLOT_REWRITE_ )
     859                   << " topslot_rewrite " << m_topslot_rewrite
     860                   << " prior_flag -> flag " <<   OpticksFlags::Abbrev(prior_flag)
     861                   << " -> " <<   OpticksFlags::Abbrev(flag)
     862                   << " prior_mat -> mat "
     863                   <<   ( prior_mat == 0 ? "-" : m_material_bridge->getMaterialName(prior_mat-1, true)  )
     864                   << " -> "
     865                   <<   ( mat == 0       ? "-" : m_material_bridge->getMaterialName(mat-1, true)  )
     866                   ;
     867 
     868         // allowing a single AB->RE rewrite is closer to Opticks
     869         if(m_topslot_rewrite == 1 && flag == BULK_REEMIT && prior_flag == BULK_ABSORB)
     870         {
     871             m_step_action |= TOPSLOT_REWRITE ;
     872         }   
     873         else
     874         {
     875             m_step_action |= HARD_TRUNCATE ;
     876             return true ; 
     877         }   
     878     }   
     ...
     881     m_seqhis =  (m_seqhis & (~msk)) | (his << shift) ;
     882     m_seqmat =  (m_seqmat & (~msk)) | (mat << shift) ;
     883     m_mskhis |= flag ;  
     884     if(flag == BULK_REEMIT) m_mskhis = m_mskhis & (~BULK_ABSORB)  ;
     ...
     897     //  Decrementing m_slot and running again will not scrub the AB from the mask
     898     //  so need to scrub the AB (BULK_ABSORB) when a RE (BULK_REEMIT) from rejoining
     899     //  occurs. 
     900     //
     901     //  Thus should always be correct as AB is a terminating flag, 
     902     //  so any REJOINed photon will have an AB in the mask
     903     //  that needs to be a RE instead.
     904     //
     905     //  What about SA/SD ... those should never REjoin ?
     906 
     907     RecordStepPoint(slot, point, flag, material, label);
     908 
     909     double time = point->GetGlobalTime();
     910 
     911 
     912     if(m_debug || m_other) Collect(point, flag, material, boundary_status, m_mskhis, m_seqhis, m_seqmat, time);
     913 
     914     m_slot += 1 ;    // m_slot is incremented regardless of truncation, only local *slot* is constrained to recording range
     915 
     916     m_bounce_truncate = m_slot > m_bounce_max  ;  
     917     if(m_bounce_truncate) m_step_action |= BOUNCE_TRUNCATE ;
     918 
     919 
     920     bool done = m_bounce_truncate || m_record_truncate || absorb ;  
     921 
     922     if(done && m_dynamic)
     923     {
     924         m_records->add(m_dynamic_records);
     925     }
     926 
     927     return done ;   
     928 }



