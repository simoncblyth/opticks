Reemission Review
====================



recsel segv : last one issue
-------------------------------

::

    (lldb) p *f
    (float) $1 = 0.000000000000000000000000000000000000000000360133705
    (lldb) p *(f+1)
    (float) $2 = 0.000000000000000000000000000000000000000000360133705
    (lldb) p count
    (unsigned int) $3 = 16000000
    (lldb) p i
    (unsigned int) $4 = 15999998
    (lldb) 

    (lldb) p *this
    (ViewNPY) $7 = {
      m_name = 0x0000000918243c70 "rsel"
      m_npy = 0x0000000918244780
      m_parent = 0x0000000000000000
      m_bytes = 0x000000091a7a2000
      m_j = '\0'
      m_k = '\0'
      m_l = '\0'
      m_size = 4
      m_type = UNSIGNED_BYTE
      m_norm = false
      m_iatt = true
      m_item_from_dim = 2
      m_numbytes = 64000000
      m_stride = 4
      m_offset = 0
      m_low = 0x0000000000000000



    (lldb) bt
    * thread #1: tid = 0x315978, 0x000000010074ab85 libNPY.dylib`glm::tvec3<float, (this=0x00007fff5fbfd0d0, a=0x000000091e4aaff8, b=0x000000091e4aaffc, c=0x000000091e4ab000)0>::tvec3(float const&, float const&, float const&) + 53 at type_vec3.inl:71, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x91e4ab000)
      * frame #0: 0x000000010074ab85 libNPY.dylib`glm::tvec3<float, (this=0x00007fff5fbfd0d0, a=0x000000091e4aaff8, b=0x000000091e4aaffc, c=0x000000091e4ab000)0>::tvec3(float const&, float const&, float const&) + 53 at type_vec3.inl:71
        frame #1: 0x0000000100745a9d libNPY.dylib`glm::tvec3<float, (this=0x00007fff5fbfd0d0, a=0x000000091e4aaff8, b=0x000000091e4aaffc, c=0x000000091e4ab000)0>::tvec3(float const&, float const&, float const&) + 45 at type_vec3.inl:71
        frame #2: 0x000000010075246d libNPY.dylib`ViewNPY::findBounds(this=0x00000009182446c0) + 317 at ViewNPY.cpp:287
        frame #3: 0x000000010075163c libNPY.dylib`ViewNPY::addressNPY(this=0x00000009182446c0) + 28 at ViewNPY.cpp:207
        frame #4: 0x0000000100751304 libNPY.dylib`ViewNPY::init(this=0x00000009182446c0) + 340 at ViewNPY.cpp:155
        frame #5: 0x000000010075116c libNPY.dylib`ViewNPY::ViewNPY(this=0x00000009182446c0, name=0x0000000100a0471b, npy=0x0000000918244780, j=0, k=0, l=0, size=4, type=UNSIGNED_BYTE, norm=false, iatt=true, item_from_dim=2) + 332 at ViewNPY.cpp:84
        frame #6: 0x00000001007513b0 libNPY.dylib`ViewNPY::ViewNPY(this=0x00000009182446c0, name=0x0000000100a0471b, npy=0x0000000918244780, j=0, k=0, l=0, size=4, type=UNSIGNED_BYTE, norm=false, iatt=true, item_from_dim=2) + 160 at ViewNPY.cpp:85
        frame #7: 0x0000000100972f2d libOpticksCore.dylib`OpticksEvent::setRecselData(this=0x000000011156c280, recsel_data=0x0000000918244780) + 189 at OpticksEvent.cc:1304
        frame #8: 0x000000010097b5ef libOpticksCore.dylib`OpticksEvent::indexPhotonsCPU(this=0x000000011156c280) + 2063 at OpticksEvent.cc:1974
        frame #9: 0x000000010097adb3 libOpticksCore.dylib`OpticksEvent::postPropagateGeant4(this=0x000000011156c280) + 499 at OpticksEvent.cc:1908
        frame #10: 0x0000000103eae635 libcfg4.dylib`CG4::postpropagate(this=0x000000010c06a920) + 997 at CG4.cc:290
        frame #11: 0x0000000103eae194 libcfg4.dylib`CG4::propagate(this=0x000000010c06a920) + 2052 at CG4.cc:271
        frame #12: 0x0000000103f8c52a libokg4.dylib`OKG4Mgr::propagate(this=0x00007fff5fbfe840) + 538 at OKG4Mgr.cc:82
        frame #13: 0x00000001000139ca OKG4Test`main(argc=27, argv=0x00007fff5fbfe920) + 1498 at OKG4Test.cc:57
        frame #14: 0x00007fff8b3ee5fd libdyld.dylib`start + 1
    (lldb) f 8
    frame #8: 0x000000010097b5ef libOpticksCore.dylib`OpticksEvent::indexPhotonsCPU(this=0x000000011156c280) + 2063 at OpticksEvent.cc:1974
       1971 
       1972     if(recsel0 && recsel0->hasData()) LOG(warning) << " leaking recsel0 " ; 
       1973 
    -> 1974     setRecselData(recsel1);
       1975 
       1976     setHistoryIndex(idx->getHistoryIndex());
       1977     setMaterialIndex(idx->getMaterialIndex());
    (lldb) 



fixing REJOIN recording
--------------------------


checking matching with extremes of material properties
---------------------------------------------------------

No surprises with extrema, so issue from interplay ?


Dial up absorption and scattering, both lengths are set to 100mm so its 50:50 which happens first, 
(summing up "TO SC xx" will get close to half) note good agreement

This almost corresponds to the infinite detector case, but not quite as the 
lengths are both set to constants... 

* TODO: support scaling of scattering/absorption lengths, so can properly do the 
  infinite detector case 


::

    tlaser-t --xxre --xxab --nosc  --bouncemax 15 --recordmax 16     ## pushing out the truncation

         seqhis_ana     1:laser     -1:laser           c2 
                  4d        499722       499926             0.04  [2 ] TO AB
                 45d        250289       249482             1.30  [3 ] TO RE AB
                455d        124565       125538             3.79  [4 ] TO RE RE AB
               4555d         62467        62510             0.01  [5 ] TO RE RE RE AB
              45555d         31409        31393             0.00  [6 ] TO RE RE RE RE AB
             455555d         15795        15711             0.22  [7 ] TO RE RE RE RE RE AB
            4555555d          7942         7837             0.70  [8 ] TO RE RE RE RE RE RE AB
           45555555d          3803         3893             1.05  [9 ] TO RE RE RE RE RE RE RE AB
          455555555d          2001         1890             3.17  [10] TO RE RE RE RE RE RE RE RE AB
         4555555555d           979          929             1.31  [11] TO RE RE RE RE RE RE RE RE RE AB
        45555555555d           534          448             7.53  [12] TO RE RE RE RE RE RE RE RE RE RE AB
       455555555555d           236          216             0.88  [13] TO RE RE RE RE RE RE RE RE RE RE RE AB
      4555555555555c           114          104             0.46  [14] BT RE RE RE RE RE RE RE RE RE RE RE RE AB
     455555555555540            71           64             0.36  [15] ?0? AB RE RE RE RE RE RE RE RE RE RE RE RE AB
    4555555555555400            36           25             1.98  [16] ?0? ?0? AB RE RE RE RE RE RE RE RE RE RE RE RE AB
    5555555555555400            35           30             0.38  [16] ?0? ?0? AB RE RE RE RE RE RE RE RE RE RE RE RE RE
              4c555d             0            2             0.00  [6 ] TO RE RE RE BT AB
               4cc5d             1            0             0.00  [5 ] TO RE BT BT AB
     4cc5cc555555540             0            1             0.00  [15] ?0? AB RE RE RE RE RE RE RE BT BT RE BT BT AB
             4c5555d             1            0             0.00  [7 ] TO RE RE RE RE BT AB
                         1000000      1000000         1.45 


    tlaser-t --xxre --xxab --nosc  

       ## pump up the stats, possibly bounce max truncation effect

         seqhis_ana     1:laser     -1:laser           c2 
                  4d        499722       499926             0.04  [2 ] TO AB
                 45d        250289       249483             1.30  [3 ] TO RE AB
                455d        124565       125538             3.79  [4 ] TO RE RE AB
               4555d         62467        62509             0.01  [5 ] TO RE RE RE AB
              45555d         31409        31393             0.00  [6 ] TO RE RE RE RE AB
             455555d         15795        15711             0.22  [7 ] TO RE RE RE RE RE AB
            4555555d          7942         7837             0.70  [8 ] TO RE RE RE RE RE RE AB
           45555555d          3803         3893             1.05  [9 ] TO RE RE RE RE RE RE RE AB
          555555555d          2005         1816             9.35  [10] TO RE RE RE RE RE RE RE RE RE
          455555555d          2001         1890             3.17  [10] TO RE RE RE RE RE RE RE RE AB
              4c555d             0            2             0.00  [6 ] TO RE RE RE BT AB
          cc5555555d             0            1             0.00  [10] TO RE RE RE RE RE RE RE BT BT
               4cc5d             1            0             0.00  [5 ] TO RE BT BT AB
             4c5555d             1            0             0.00  [7 ] TO RE RE RE RE BT AB
          c55555555d             0            1             0.00  [10] TO RE RE RE RE RE RE RE RE BT
                         1000000      1000000         1.96 
          seqmat_ana     1:laser     -1:laser           c2 
                  11        499722       499926             0.04  [2 ] Gd Gd
                 111        250289       249483             1.30  [3 ] Gd Gd Gd
                1111        124565       125538             3.79  [4 ] Gd Gd Gd Gd
               11111         62467        62509             0.01  [5 ] Gd Gd Gd Gd Gd
              111111         31409        31393             0.00  [6 ] Gd Gd Gd Gd Gd Gd
             1111111         15795        15711             0.22  [7 ] Gd Gd Gd Gd Gd Gd Gd
            11111111          7942         7837             0.70  [8 ] Gd Gd Gd Gd Gd Gd Gd Gd
          1111111111          4006         3706            11.67  [10] Gd Gd Gd Gd Gd Gd Gd Gd Gd Gd
           111111111          3803         3893             1.05  [9 ] Gd Gd Gd Gd Gd Gd Gd Gd Gd
              331111             0            2             0.00  [6 ] Gd Gd Gd Gd Ac Ac
          2311111111             0            1             0.00  [10] Gd Gd Gd Gd Gd Gd Gd Gd Ac LS
             3311111             1            0             0.00  [7 ] Gd Gd Gd Gd Gd Ac Ac
               33211             1            0             0.00  [5 ] Gd Gd LS Ac Ac
          3111111111             0            1             0.00  [10] Gd Gd Gd Gd Gd Gd Gd Gd Gd Ac
                         1000000      1000000         2.09 



        ## finally rejoin logic yields more sensical seqs 

          seqhis_ana     1:laser     -1:laser           c2 
                  4d            48           44             0.17  [2 ] TO AB
                 45d            29           31             0.07  [3 ] TO RE AB
                455d            10           15             0.00  [4 ] TO RE RE AB
              45555d             5            1             0.00  [6 ] TO RE RE RE RE AB
               4555d             5            5             0.00  [5 ] TO RE RE RE AB
            4555555d             1            2             0.00  [8 ] TO RE RE RE RE RE RE AB
             455555d             2            2             0.00  [7 ] TO RE RE RE RE RE AB
                             100          100         0.12 

    tlaser-d --xxre --xxab --nosc  

         ## hmm, getting an inkling of cause of REJOIN problem
         ##
         ## in reality are seeing a new trk (most often "TO AB") from Scintillation with primary 
         ## matching a preexisting record ... so can rejoin 
         ## (but that is not a single entry but rather tis both a REJOIN:RE and a RECOLL:AB )
         ##
         ## hmm should be an AB and only when a REJOIN track comes along can know it was actually a RE

          seqhis_ana     1:laser     -1:laser           c2 
                  4d            48           44             0.17  [2 ] TO AB
                  5d             0           31            31.00  [2 ] TO RE
                 45d            29            0             0.00  [3 ] TO RE AB
                 55d             0           15             0.00  [3 ] TO RE RE
                455d            10            0             0.00  [4 ] TO RE RE AB
                555d             0            5             0.00  [4 ] TO RE RE RE
              45555d             5            0             0.00  [6 ] TO RE RE RE RE AB
               4555d             5            0             0.00  [5 ] TO RE RE RE AB
              55555d             0            2             0.00  [6 ] TO RE RE RE RE RE
             555555d             0            2             0.00  [7 ] TO RE RE RE RE RE RE
             455555d             2            0             0.00  [7 ] TO RE RE RE RE RE AB
            4555555d             1            0             0.00  [8 ] TO RE RE RE RE RE RE AB
               5555d             0            1             0.00  [5 ] TO RE RE RE RE


    tlaser-d --xxre --xxab --nosc     ## dial down stats for --steppingdbg,   using 50% reemission_prob
        
       ## tis better debug environment that searching for diffs in 35/1000000
       ##   huh the dumped seqhis is entirely different to the index ???  SMOKING GUN BUG 

         seqhis_ana     1:laser     -1:laser           c2 
                  4d            48          100            18.27  [2 ] TO AB
                 45d            29            0             0.00  [3 ] TO RE AB
                455d            10            0             0.00  [4 ] TO RE RE AB
               4555d             5            0             0.00  [5 ] TO RE RE RE AB
              45555d             5            0             0.00  [6 ] TO RE RE RE RE AB
             455555d             2            0             0.00  [7 ] TO RE RE RE RE RE AB
            4555555d             1            0             0.00  [8 ] TO RE RE RE RE RE RE AB



    tlaser-t --xxre --xxab --nosc
    tlaser-t --xxre --xxab --nosc --dbgseqhis 45d   ## hmm dbgseqhis uses the CFG4 seqhis so not so useful

      ## switch off SC to see bug more clearly, looks to be halfing for each category   

          seqhis_ana     1:laser     -1:laser           c2 
                  4d         49755        99819         16756.95  [2 ] TO AB
                 45d         25261            0         25261.00  [3 ] TO RE AB
                455d         12349            0         12349.00  [4 ] TO RE RE AB
               4555d          6223            0          6223.00  [5 ] TO RE RE RE AB
              45555d          3204            0          3204.00  [6 ] TO RE RE RE RE AB
             455555d          1602            0          1602.00  [7 ] TO RE RE RE RE RE AB
            4555555d           791            0           791.00  [8 ] TO RE RE RE RE RE RE AB
           45555555d           391            0           391.00  [9 ] TO RE RE RE RE RE RE RE AB
          455555555d           213            0           213.00  [10] TO RE RE RE RE RE RE RE RE AB
          555555555d           211          180             2.46  [10] TO RE RE RE RE RE RE RE RE RE
          cc5555555d             0            1             0.00  [10] TO RE RE RE RE RE RE RE BT BT
                          100000       100000      6679.34 
          seqmat_ana     1:laser     -1:laser           c2 
                  11         49755        99819         16756.95  [2 ] Gd Gd
                 111         25261            0         25261.00  [3 ] Gd Gd Gd
                1111         12349            0         12349.00  [4 ] Gd Gd Gd Gd
               11111          6223            0          6223.00  [5 ] Gd Gd Gd Gd Gd
              111111          3204            0          3204.00  [6 ] Gd Gd Gd Gd Gd Gd
             1111111          1602            0          1602.00  [7 ] Gd Gd Gd Gd Gd Gd Gd
            11111111           791            0           791.00  [8 ] Gd Gd Gd Gd Gd Gd Gd Gd
          1111111111           424          180            98.57  [10] Gd Gd Gd Gd Gd Gd Gd Gd Gd Gd
           111111111           391            0           391.00  [9 ] Gd Gd Gd Gd Gd Gd Gd Gd Gd
          2311111111             0            1             0.00  [10] Gd Gd Gd Gd Gd Gd Gd Gd Ac LS
                          100000       100000      7408.50 


    tlaser-t --xxre --xxab --xxsc   

      ## with xxre corresponding to 50% reemission_prob
      ## CFG4 not yielding any TO RE AB  (must be a REJOIN bug ?)

         seqhis_ana     1:laser     -1:laser           c2 
                  4d         24925        33517          1263.17  [2 ] TO AB
                 46d         12509        16747           613.91  [3 ] TO SC AB
                466d          6246         8262           280.14  [4 ] TO SC SC AB
                 45d          6326            0          6326.00  [3 ] TO RE AB          <--- REJOIN bug ? no reason for this not to happen ?
                465d          3140         4250           166.73  [4 ] TO RE SC AB
               4666d          3039         4170           177.44  [5 ] TO SC SC SC AB
                456d          3123            0          3123.00  [4 ] TO SC RE AB       <-- REJOIN bug ?  
               4665d          1563         2143            90.77  [5 ] TO RE SC SC AB
              46666d          1564         2114            82.25  [6 ] TO SC SC SC SC AB
               4656d          1541         2102            86.39  [5 ] TO SC RE SC AB
               4566d          1637            0          1637.00  [5 ] TO SC SC RE AB    <-- REJOIN bug ?  
                455d          1541            0          1541.00  [4 ] TO RE RE AB     
              46656d           796         1081            43.27  [6 ] TO SC RE SC SC AB
              46566d           767         1048            43.50  [6 ] TO SC SC RE SC AB
             466666d           786         1030            32.78  [7 ] TO SC SC SC SC SC AB
               4655d           765          994            29.81  [5 ] TO RE RE SC AB
              46665d           797          955            14.25  [6 ] TO RE SC SC SC AB
              45666d           856            0           856.00  [6 ] TO SC SC SC RE AB
               4556d           839            0           839.00  [5 ] TO SC RE RE AB
               4565d           773            0           773.00  [5 ] TO RE SC RE AB


    ## hmm 100% reemission_prob is too unphysical, need to leave some possibility of AB otherwise 
    ## just truncate and get no sensible sequence index

          seqhis_ana     1:laser     -1:laser           c2 
          556566665d           207          336            30.65  [10] TO RE SC SC SC SC RE SC RE RE
          565666655d           197          332            34.45  [10] TO RE RE SC SC SC SC RE SC RE
          566655566d           167          331            54.01  [10] TO SC SC RE RE RE SC SC SC RE
          556655565d           178          330            45.48  [10] TO RE SC RE RE RE SC SC RE RE
          555566656d           183          329            41.63  [10] TO SC RE SC SC SC RE RE RE RE
          565665556d           169          329            51.41  [10] TO SC RE RE RE SC SC RE SC RE
          566655565d           203          325            28.19  [10] TO RE SC RE RE RE SC SC SC RE
          556556566d           188          325            36.59  [10] TO SC SC RE SC RE RE SC RE RE
          556555566d           200          325            29.76  [10] TO SC SC RE RE RE RE SC RE RE
          556556665d           171          323            46.77  [10] TO RE SC SC SC RE RE SC RE RE
          565566555d           194          322            31.75  [10] TO RE RE RE SC SC RE RE SC RE
          555556555d           192          322            32.88  [10] TO RE RE RE SC RE RE RE RE RE
          555566566d           195          318            29.49  [10] TO SC SC RE SC SC RE RE RE RE
          556666656d           222          317            16.74  [10] TO SC RE SC SC SC SC SC RE RE
          565665565d           194          317            29.61  [10] TO RE SC RE RE SC SC RE SC RE
          555556656d           193          317            30.15  [10] TO SC RE SC SC RE RE RE RE RE
          555555555d           213          316            20.05  [10] TO RE RE RE RE RE RE RE RE RE
          556556555d           192          316            30.27  [10] TO RE RE RE SC RE RE SC RE RE
          556556655d           187          316            33.08  [10] TO RE RE SC SC RE RE SC RE RE
          556556556d           179          316            37.92  [10] TO SC RE RE SC RE RE SC RE RE





::

    tlaser-t --nore --xxab --xxsc

          seqhis_ana     1:laser     -1:laser           c2 
                  4d         49958        50039             0.07  [2 ] TO AB
                 46d         25055        24984             0.10  [3 ] TO SC AB
                466d         12502        12496             0.00  [4 ] TO SC SC AB
               4666d          6236         6246             0.01  [5 ] TO SC SC SC AB
              46666d          3117         3079             0.23  [6 ] TO SC SC SC SC AB
             466666d          1547         1607             1.14  [7 ] TO SC SC SC SC SC AB
            4666666d           818          757             2.36  [8 ] TO SC SC SC SC SC SC AB
           46666666d           373          381             0.08  [9 ] TO SC SC SC SC SC SC SC AB
          466666666d           204          209             0.06  [10] TO SC SC SC SC SC SC SC SC AB
          666666666d           190          202             0.37  [10] TO SC SC SC SC SC SC SC SC SC
                          100000       100000         0.44 
          seqmat_ana     1:laser     -1:laser           c2 
                  11         49958        50039             0.07  [2 ] Gd Gd
                 111         25055        24984             0.10  [3 ] Gd Gd Gd
                1111         12502        12496             0.00  [4 ] Gd Gd Gd Gd
               11111          6236         6246             0.01  [5 ] Gd Gd Gd Gd Gd
              111111          3117         3079             0.23  [6 ] Gd Gd Gd Gd Gd Gd
             1111111          1547         1607             1.14  [7 ] Gd Gd Gd Gd Gd Gd Gd
            11111111           818          757             2.36  [8 ] Gd Gd Gd Gd Gd Gd Gd Gd
          1111111111           394          411             0.36  [10] Gd Gd Gd Gd Gd Gd Gd Gd Gd Gd
           111111111           373          381             0.08  [9 ] Gd Gd Gd Gd Gd Gd Gd Gd Gd



Dialing up absorption and reemission runs slow::

       tlaser-t --xxab --nosc --xxre 

         seqhis_ana     1:laser     -1:laser           c2 
          555555555d         99979        99985             0.00  [10] TO RE RE RE RE RE RE RE RE RE
          c55555555d            12            7             0.00  [10] TO RE RE RE RE RE RE RE RE BT
          5c5555555d             0            5             0.00  [10] TO RE RE RE RE RE RE RE BT RE
          cc5555555d             2            0             0.00  [10] TO RE RE RE RE RE RE RE BT BT
          55cc55555d             2            1             0.00  [10] TO RE RE RE RE RE BT BT RE RE
          555cc5555d             2            0             0.00  [10] TO RE RE RE RE BT BT RE RE RE
          5cc555555d             2            0             0.00  [10] TO RE RE RE RE RE RE BT BT RE
          5ccc55555d             1            0             0.00  [10] TO RE RE RE RE RE BT BT BT RE
           c5555555d             0            1             0.00  [9 ] TO RE RE RE RE RE RE RE BT
          55555cc55d             0            1             0.00  [10] TO RE RE BT BT RE RE RE RE RE
                          100000       100000         0.00 
          seqmat_ana     1:laser     -1:laser           c2 
          1111111111         99991        99985             0.00  [10] Gd Gd Gd Gd Gd Gd Gd Gd Gd Gd
          3111111111             0            7             0.00  [10] Gd Gd Gd Gd Gd Gd Gd Gd Gd Ac
          2311111111             0            5             0.00  [10] Gd Gd Gd Gd Gd Gd Gd Gd Ac LS
          3311111111             2            1             0.00  [10] Gd Gd Gd Gd Gd Gd Gd Gd Ac Ac
          2222311111             2            0             0.00  [10] Gd Gd Gd Gd Gd Ac LS LS LS LS
          2231111111             2            0             0.00  [10] Gd Gd Gd Gd Gd Gd Gd Ac LS LS
          2223111111             2            1             0.00  [10] Gd Gd Gd Gd Gd Gd Ac LS LS LS
          2232111111             1            0             0.00  [10] Gd Gd Gd Gd Gd Gd LS Ac LS LS
          2222223111             0            1             0.00  [10] Gd Gd Gd Ac LS LS LS LS LS LS

Dialing up reemission alone doesnt produce any RE as its handled as subset of AB::

       tlaser-t --noab --nosc --xxre 

         seqhis_ana     1:laser     -1:laser           c2 
              8ccccd         95962        95937             0.00  [6 ] TO BT BT BT BT SA
          cccc9ccccd          3449         3460             0.02  [10] TO BT BT BT BT DR BT BT BT BT
             89ccccd           265          293             1.41  [7 ] TO BT BT BT BT DR SA
            8c9ccccd            42           88            16.28  [8 ] TO BT BT BT BT DR BT SA
           ccc9ccccd             0           75            75.00  [9 ] TO BT BT BT BT DR BT BT BT
            8b9ccccd            36           18             6.00  [8 ] TO BT BT BT BT DR BR SA
          bbbb9ccccd            36            0            36.00  [10] TO BT BT BT BT DR BR BR BR BR
          bccc9ccccd            32           31             0.02  [10] TO BT BT BT BT DR BT BT BT BR
          ccbc9ccccd            25            3             0.00  [10] TO BT BT BT BT DR BT BR BT BT
          cacc9ccccd            11           24             4.83  [10] TO BT BT BT BT DR BT BT SR BT
            7c9ccccd             7           23             0.00  [8 ] TO BT BT BT BT DR BT SD
          8cbb9ccccd            18            0             0.00  [10] TO BT BT BT BT DR BR BR BT SA
           8bb9ccccd            18            0             0.00  [9 ] TO BT BT BT BT DR BR BR SA
          ccc99ccccd            14            8             0.00  [10] TO BT BT BT BT DR DR BT BT BT
           8cc9ccccd            10           11             0.00  [9 ] TO BT BT BT BT DR BT BT SA
          cccccbcccd            10            8             0.00  [10] TO BT BT BT BR BT BT BT BT BT
          8bbb9ccccd             9            0             0.00  [10] TO BT BT BT BT DR BR BR BR SA
           8cb9ccccd             7            1             0.00  [9 ] TO BT BT BT BT DR BR BT SA
          8cbc9ccccd             5            5             0.00  [10] TO BT BT BT BT DR BT BR BT SA
            899ccccd             5            1             0.00  [8 ] TO BT BT BT BT DR DR SA

With scattering dialed up::

   tlaser-t --noab --xxsc --nore 

         seqhis_ana     1:laser     -1:laser           c2 
          666666666d         99985        99991             0.00  [10] TO SC SC SC SC SC SC SC SC SC
          c66666666d             6            6             0.00  [10] TO SC SC SC SC SC SC SC SC BT
          cc6666666d             5            1             0.00  [10] TO SC SC SC SC SC SC SC BT BT
          6cc666666d             0            2             0.00  [10] TO SC SC SC SC SC SC BT BT SC
          bc6666666d             1            0             0.00  [10] TO SC SC SC SC SC SC SC BT BR
          6c6666666d             1            0             0.00  [10] TO SC SC SC SC SC SC SC BT SC
          66cc66666d             1            0             0.00  [10] TO SC SC SC SC SC BT BT SC SC
          66c66cc66d             1            0             0.00  [10] TO SC SC BT BT SC SC BT SC SC
                          100000       100000         0.00 
          seqmat_ana     1:laser     -1:laser           c2 
          1111111111         99991        99991             0.00  [10] Gd Gd Gd Gd Gd Gd Gd Gd Gd Gd
          3311111111             6            0             0.00  [10] Gd Gd Gd Gd Gd Gd Gd Gd Ac Ac
          3111111111             0            6             0.00  [10] Gd Gd Gd Gd Gd Gd Gd Gd Gd Ac
          2231111111             0            2             0.00  [10] Gd Gd Gd Gd Gd Gd Gd Ac LS LS
          2311111111             0            1             0.00  [10] Gd Gd Gd Gd Gd Gd Gd Gd Ac LS
          2211111111             1            0             0.00  [10] Gd Gd Gd Gd Gd Gd Gd Gd LS LS
          2223111111             1            0             0.00  [10] Gd Gd Gd Gd Gd Gd Ac LS LS LS
          2223332111             1            0             0.00  [10] Gd Gd Gd LS Ac Ac Ac LS LS LS


With absorption dialed up, nothing else has a chance::

    tlaser-t --xxab --nosc --nore 

         seqhis_ana     1:laser     -1:laser           c2 
                  4d        100000       100000             0.00  [2 ] TO AB
                          100000       100000         0.00 
          seqmat_ana     1:laser     -1:laser           c2 
                  11        100000       100000             0.00  [2 ] Gd Gd
                          100000       100000         0.00 



Killing AB SC (with 1 million mm scattering and absorption lengths) and RE (prob 0) 
leaves just boundary processes with agree quite well (not the same geometry due
to triangulation so some low level discrep to be expected)::

    simon:ggeo blyth$ tlaser-t --noab --nosc --nore 


         seqhis_ana     1:laser     -1:laser           c2 
              8ccccd         95535        95346             0.19  [6 ] TO BT BT BT BT SA
          cccc9ccccd          3424         3579             3.43  [10] TO BT BT BT BT DR BT BT BT BT
             89ccccd           263          295             1.84  [7 ] TO BT BT BT BT DR SA
                  4d           154          166             0.45  [2 ] TO AB
             8cccc6d           111           86             3.17  [7 ] TO SC BT BT BT BT SA
            8c9ccccd            41           95            21.44  [8 ] TO BT BT BT BT DR BT SA
           ccc9ccccd             0           93            93.00  [9 ] TO BT BT BT BT DR BT BT BT
          cacc9ccccd            11           36            13.30  [10] TO BT BT BT BT DR BT BT SR BT
          bbbb9ccccd            36            0            36.00  [10] TO BT BT BT BT DR BR BR BR BR
            8b9ccccd            36           13            10.80  [8 ] TO BT BT BT BT DR BR SA
                4ccd            32           34             0.06  [4 ] TO BT BT AB
          bccc9ccccd            32           23             1.47  [10] TO BT BT BT BT DR BT BT BT BR
              4ccccd            22           29             0.96  [6 ] TO BT BT BT BT AB
          ccbc9ccccd            25            3             0.00  [10] TO BT BT BT BT DR BT BR BT BT
            7c9ccccd             7           25            10.12  [8 ] TO BT BT BT BT DR BT SD
             8cc6ccd            19           21             0.10  [7 ] TO BT BT SC BT BT SA
          cccccc6ccd            20           14             1.06  [10] TO BT BT SC BT BT BT BT BT BT
           8bb9ccccd            18            0             0.00  [9 ] TO BT BT BT BT DR BR BR SA
          8cbb9ccccd            18            0             0.00  [10] TO BT BT BT BT DR BR BR BT SA
          cacccccc6d            14           14             0.00  [10] TO SC BT BT BT BT BT BT SR BT



switch off reemission with nore option
-----------------------------------------

Now with tex buffer updated both do no RE, note very different AB BULK_ABSORB::



     tlaser-t --nore

         seqhis_ana     1:laser     -1:laser           c2 
              8ccccd         76521        81497           156.69  [6 ] TO BT BT BT BT SA
                  4d         11030         7100           851.90  [2 ] TO AB               ## Opticks AB much more ???
          cccc9ccccd          2428         2701            14.53  [10] TO BT BT BT BT DR BT BT BT BT
                4ccd          2433         1695           131.94  [4 ] TO BT BT AB
             8cccc6d          1980         1847             4.62  [7 ] TO SC BT BT BT BT SA
              4ccccd           822          915             4.98  [6 ] TO BT BT BT BT AB
                 46d           428          246            49.15  [3 ] TO SC AB
             8cc6ccd           413          402             0.15  [7 ] TO BT BT SC BT BT SA
             86ccccd           299          275             1.00  [7 ] TO BT BT BT BT SC SA
          cccccc6ccd           262          185            13.26  [10] TO BT BT SC BT BT BT BT BT BT
          cccc6ccccd           229          164            10.75  [10] TO BT BT BT BT SC BT BT BT BT
          cacccccc6d           205          215             0.24  [10] TO SC BT BT BT BT BT BT SR BT
               4cccd           209          196             0.42  [5 ] TO BT BT BT AB
             89ccccd           191          197             0.09  [7 ] TO BT BT BT BT DR SA
            8ccccc6d           122          175             9.46  [8 ] TO SC BT BT BT BT BT SA
            4ccccc6d           152           11           121.97  [8 ] TO SC BT BT BT BT BT AB
          ccbccccc6d           135          143             0.23  [10] TO SC BT BT BT BT BT BR BT BT
           4cc9ccccd           140          114             2.66  [9 ] TO BT BT BT BT DR BT BT AB
           cac0ccc6d             0          134           134.00  [9 ] TO SC BT BT BT ?0? BT SR BT
               4cc6d           128           76            13.25  [5 ] TO SC BT BT AB


Before updated the tex buffer opticks still doing RE::

         seqhis_ana     1:laser     -1:laser           c2 
              8ccccd         76521        81497           156.69  [6 ] TO BT BT BT BT SA
                  4d          5573         7100           183.99  [2 ] TO AB
          cccc9ccccd          2428         2701            14.53  [10] TO BT BT BT BT DR BT BT BT BT
             8cccc6d          1980         1847             4.62  [7 ] TO SC BT BT BT BT SA
                4ccd          1194         1695            86.88  [4 ] TO BT BT AB
             8cccc5d          1074            0          1074.00  [7 ] TO RE BT BT BT BT SA
              4ccccd           822          915             4.98  [6 ] TO BT BT BT BT AB
                 45d           754            0           754.00  [3 ] TO RE AB
            8cccc55d           561            0           561.00  [8 ] TO RE RE BT BT BT BT SA
             8cc6ccd           413          402             0.15  [7 ] TO BT BT SC BT BT SA
                455d           345            0           345.00  [4 ] TO RE RE AB
             86ccccd           299          275             1.00  [7 ] TO BT BT BT BT SC SA
          cccccc6ccd           262          185            13.26  [10] TO BT BT SC BT BT BT BT BT BT
                 46d           217          246             1.82  [3 ] TO SC AB
           8cccc555d           243            0           243.00  [9 ] TO RE RE RE BT BT BT BT SA
             8cc5ccd           236            0           236.00  [7 ] TO BT BT RE BT BT SA
          cccc6ccccd           229          164            10.75  [10] TO BT BT BT BT SC BT BT BT BT
          cacccccc6d           205          215             0.24  [10] TO SC BT BT BT BT BT BT SR BT
               4cccd           209          196             0.42  [5 ] TO BT BT BT AB
             89ccccd           191          197             0.09  [7 ] TO BT BT BT BT DR SA
                          100000       100000        79.91 





tlaser : sizable differences in many categories : how to proceed ?
---------------------------------------------------------------------

Ideas to isolate the issue:

* switch off reemission, and compare without it 
* arrange effectively infinite sphere of scintillator and try tlaser in that  

* suspect difference in multi-reemission 
* sequence recording bugs regards the reemission are also possible 

::

        After fixing REJOIN issue, some zeroes removed, but still big discreps:

         seqhis_ana     1:laser     -1:laser           c2 
              8ccccd        763501       813497          1585.04  [6 ] TO BT BT BT BT SA
                  4d         55825        47634           648.49  [2 ] TO AB
          cccc9ccccd         25263        26200            17.06  [10] TO BT BT BT BT DR BT BT BT BT
             8cccc6d         19707        18533            36.04  [7 ] TO SC BT BT BT BT SA
                4ccd         12576        11563            42.51  [4 ] TO BT BT AB
             8cccc5d         11183         7742           625.65  [7 ] TO RE BT BT BT BT SA
              4ccccd          8554         8756             2.36  [6 ] TO BT BT BT BT AB
                 45d          7531         2208          2909.37  [3 ] TO RE AB
            8cccc55d          5362         2116          1409.00  [8 ] TO RE RE BT BT BT BT SA
             8cc6ccd          4109         4155             0.26  [7 ] TO BT BT SC BT BT SA
                455d          3588          621          2091.49  [4 ] TO RE RE AB
             86ccccd          2836         2743             1.55  [7 ] TO BT BT BT BT SC SA
          cccccc6ccd          2674         1919           124.11  [10] TO BT BT SC BT BT BT BT BT BT
           8cccc555d          2524          610          1168.92  [9 ] TO RE RE RE BT BT BT BT SA
             8cc5ccd          2359         1866            57.53  [7 ] TO BT BT RE BT BT SA
             89ccccd          1880         2221            28.35  [7 ] TO BT BT BT BT DR SA
          cacccccc6d          2210         2127             1.59  [10] TO SC BT BT BT BT BT BT SR BT
                 46d          2118         1569            81.75  [3 ] TO SC AB
          cccc6ccccd          2060         1752            24.89  [10] TO BT BT BT BT SC BT BT BT BT
               4cccd          1940         1981             0.43  [5 ] TO BT BT BT AB
                         1000000      1000000       106.82 


         seqhis_ana     1:laser     -1:laser           c2 
              8ccccd         76521        81336           146.87  [6 ] TO BT BT BT BT SA
                  4d          5573         5002            30.83  [2 ] TO AB
          cccc9ccccd          2428         2661            10.67  [10] TO BT BT BT BT DR BT BT BT BT
             8cccc6d          1980         1899             1.69  [7 ] TO SC BT BT BT BT SA
                4ccd          1194         1208             0.08  [4 ] TO BT BT AB
             8cccc5d          1074          753            56.40  [7 ] TO RE BT BT BT BT SA
              4ccccd           822          858             0.77  [6 ] TO BT BT BT BT AB
                 45d           754            0           754.00  [3 ] TO RE AB
            8cccc55d           561          230           138.51  [8 ] TO RE RE BT BT BT BT SA
             8cc6ccd           413          403             0.12  [7 ] TO BT BT SC BT BT SA
                455d           345            0           345.00  [4 ] TO RE RE AB
             86ccccd           299          263             2.31  [7 ] TO BT BT BT BT SC SA
          cccccc6ccd           262          198             8.90  [10] TO BT BT SC BT BT BT BT BT BT
           8cccc555d           243           66           101.39  [9 ] TO RE RE RE BT BT BT BT SA
             8cc5ccd           236          190             4.97  [7 ] TO BT BT RE BT BT SA
          cccc6ccccd           229          164            10.75  [10] TO BT BT BT BT SC BT BT BT BT
             89ccccd           191          218             1.78  [7 ] TO BT BT BT BT DR SA
                 46d           217          148            13.04  [3 ] TO SC AB
               4cccd           209          207             0.01  [5 ] TO BT BT BT AB
          cacccccc6d           205          208             0.02  [10] TO SC BT BT BT BT BT BT SR BT
                          100000       100000        37.61 
          seqmat_ana     1:laser     -1:laser           c2 
              343231         76521        81336           146.87  [6 ] Gd Ac LS Ac MO Ac
                  11          5573         5002            30.83  [2 ] Gd Gd
             3432311          2949         2591            23.13  [7 ] Gd Gd Ac LS Ac MO Ac
          3323443231          2656          102          2365.09  [10] Gd Ac LS Ac MO MO Ac LS Ac Ac
          1323443231             0         2191          2191.00  [10] Gd Ac LS Ac MO MO Ac LS Ac Gd
                2231          1194         1208             0.08  [4 ] Gd Ac LS LS
                 111           971          148           605.30  [3 ] Gd Gd Gd
              443231           822          858             0.77  [6 ] Gd Ac LS Ac MO MO
            34323111           682          335           118.40  [8 ] Gd Gd Gd Ac LS Ac MO Ac
          4323443231             0          664           664.00  [10] Gd Ac LS Ac MO MO Ac LS Ac MO
             3432231           638          586             2.21  [7 ] Gd Ac LS LS Ac MO Ac
             3443231           475          473             0.00  [7 ] Gd Ac LS Ac MO MO Ac
          fff3432311           398           30           316.41  [10] Gd Gd Ac LS Ac MO Ac Ai Ai Ai
                1111           377           14           337.01  [4 ] Gd Gd Gd Gd
            5e432311             0          357           357.00  [8 ] Gd Gd Ac LS Ac MO Py Bk
          3323132231           350           49           227.07  [10] Gd Ac LS LS Ac Gd Ac LS Ac Ac
          3ff3432311             0          330           330.00  [10] Gd Gd Ac LS Ac MO Ac Ai Ai Ac
           343231111           294           85           115.25  [9 ] Gd Gd Gd Gd Ac LS Ac MO Ac
          3433432311            76          267           106.36  [10] Gd Gd Ac LS Ac MO Ac Ac MO Ac
          4323132231             0          265           265.00  [10] Gd Ac LS LS Ac Gd Ac LS Ac MO




push stats to 1M have 35 CRecorder/Rec discrepant seqhis/seqmat
----------------------------------------------------------------

* decided to throwaway Rec sequencing, keeping two very different 
  CFG4 implementations matched turns out to be too much work for the benefit, 
  it was distracting from primary task of matching Opticks to G4 
 

* most common discrep is, one less "c" in rec

::

    2016-10-21 13:11:34.267 INFO  [2947727] [CSteppingAction::report@380] CG4::postpropagate
     event_total 100
     track_total 1045143
     step_total 5165738
    2016-10-21 13:11:34.267 INFO  [2947727] [CRecorder::report@894] CG4::postpropagate
    2016-10-21 13:11:34.267 INFO  [2947727] [CRecorder::report@898]  seqhis_mismatch 35
     rdr           8ccccd rec            8cccd
     rdr          8c0cccd rec           8c0ccd
     rdr                d rec               4d
     rdr                d rec               4d
     rdr           8ccccd rec            8cccd
     rdr           8ccccd rec            8cccd
     rdr           8ccccd rec            8cccd
     rdr           8ccccd rec            8cccd
     rdr                d rec               4d
     rdr       c0cac0cccd rec       cc0cac0ccd
     rdr              b6d rec             4b6d
     rdr                d rec               4d
     rdr                d rec               4d
     rdr                d rec               4d
     rdr                d rec               4d
     rdr       ccaccccccd rec       cccacccccd
     rdr                d rec               4d
     rdr           8ccccd rec            8cccd
     rdr       c0cac0cccd rec       cc0cac0ccd
     rdr                d rec               4d
     rdr           8ccccd rec            8cccd
     rdr       cccc9ccccd rec       ccccc9cccd
     rdr       ccaccccccd rec       cccacccccd
     rdr                d rec               4d
     rdr         8ccccb5d rec           8ccccd
     rdr           8ccccd rec            8cccd
     rdr           8ccccd rec            8cccd
     rdr           8ccccd rec            8cccd
     rdr                d rec               4d
     rdr       c0b0cccccd rec       cc0b0ccccd
     rdr       cc0b00cccd rec       ccc0b00ccd
     rdr       cccbcccccd rec       ccccbccccd
     rdr       cacccc5ccd rec       ccacccc5cd
     rdr           8ccccd rec            8cccd
     rdr           8ccccd rec            8cccd
    2016-10-21 13:11:34.267 INFO  [2947727] [CRecorder::report@912]  seqmat_mismatch 35
     rdr           343231 rec            34323 rdr GdDopedLS Acrylic LiquidScintillator Acrylic MineralOil Acrylic - - - - - - - - - -  rec Acrylic LiquidScintillator Acrylic MineralOil Acrylic - - - - - - - - - - - 
     rdr          af33231 rec           af3323 rdr GdDopedLS Acrylic LiquidScintillator Acrylic Acrylic Air ESR - - - - - - - - -  rec Acrylic LiquidScintillator Acrylic Acrylic Air ESR - - - - - - - - - - 

* approx half have a skipped decrementSlot warning 

::

    2016-10-21 13:41:23.927 INFO  [2954706] [CSteppingAction::setEvent@179] CSA (startEvent) event_id 6 event_total 6
    2016-10-21 13:41:24.381 INFO  [2954706] [CRecorder::RecordStepPoint@576] CRecorder::RecordStepPoint m_slot 1 slot 0 flag d done N truncate N     START evt       6 pho     626 par      -1 pri 2147483647 ste    0 rid 60626 slt    1 pre     0.1 pst 2.80399 STATIC 
    2016-10-21 13:41:24.381 INFO  [2954706] [CRecorder::RecordStepPoint@576] CRecorder::RecordStepPoint m_slot 2 slot 1 flag 4 done Y truncate N     START evt       6 pho     626 par      -1 pri 2147483647 ste    0 rid 60626 slt    2 pre     0.1 pst 2.80399 STATIC 
    2016-10-21 13:41:24.381 WARN  [2954706] [CRecorder::decrementSlot@363] CRecorder::decrementSlot SKIPPING slot 0 truncate 0
    2016-10-21 13:41:24.381 INFO  [2954706] [CRecorder::RecordStepPoint@576] CRecorder::RecordStepPoint m_slot 1 slot 0 flag d done N truncate N    RECOLL evt       6 pho     626 par   10432 pri 2147483647 ste    1 rid 60626 slt    1 pre 11.0342 pst 11.0921 STATIC 
    2016-10-21 13:41:24.381 INFO  [2954706] [CRecorder::RecordStepPoint@576] CRecorder::RecordStepPoint m_slot 2 slot 1 flag c done N truncate N    RECOLL evt       6 pho     626 par   10432 pri 2147483647 ste    2 rid 60626 slt    2 pre 11.0921 pst  13.488 STATIC 
    2016-10-21 13:41:24.381 INFO  [2954706] [CRecorder::RecordStepPoint@576] CRecorder::RecordStepPoint m_slot 3 slot 2 flag c done N truncate N    RECOLL evt       6 pho     626 par   10432 pri 2147483647 ste    3 rid 60626 slt    3 pre  13.488 pst 13.5877 STATIC 
    2016-10-21 13:41:24.381 INFO  [2954706] [CRecorder::RecordStepPoint@576] CRecorder::RecordStepPoint m_slot 4 slot 3 flag c done N truncate N    RECOLL evt       6 pho     626 par   10432 pri 2147483647 ste    4 rid 60626 slt    4 pre 13.5877 pst 15.0218 STATIC 
    2016-10-21 13:41:24.381 INFO  [2954706] [CRecorder::RecordStepPoint@576] CRecorder::RecordStepPoint m_slot 5 slot 4 flag 8 done Y truncate N    RECOLL evt       6 pho     626 par   10432 pri 2147483647 ste    4 rid 60626 slt    5 pre 13.5877 pst 15.0218 STATIC 




seqhis machinery inconsistency between CRecorder and Rec
----------------------------------------------------------

::

    simon:geant4_opticks_integration blyth$ t tlaser-d
    tlaser-d () 
    { 
        tlaser-;
        tlaser-t --steppingdbg   ## dumps every event 
    }
    simon:geant4_opticks_integration blyth$ t tlaser-t
    tlaser-t () 
    { 
        tlaser-;
        tlaser-- --okg4 --compute $*
    }



CRecorder and Rec are disagreeing for the last slot at the 6 in 10k level. 
Presumably a truncation behavior difference::

    2016-10-20 11:23:58.951 INFO  [2770241] [OpticksEvent::collectPhotonHitsCPU@1924] OpticksEvent::collectPhotonHitsCPU numHits 13
    2016-10-20 11:23:58.951 INFO  [2770241] [CSteppingAction::report@397] CG4::postpropagate
     event_total 1
     track_total 10468
     step_total 51335
    2016-10-20 11:23:58.951 INFO  [2770241] [CSteppingAction::report@407]  seqhis_mismatch 6
     rdr       cccc9ccccd rec       5ccc9ccccd
     rdr       cccc9ccccd rec       5ccc9ccccd
     rdr       cccc9ccccd rec       5ccc9ccccd
     rdr       cccc9ccccd rec       5ccc9ccccd
     rdr       cccc9ccccd rec       5ccc9ccccd
     rdr       cccc9ccccd rec       5ccc9ccccd
    2016-10-20 11:23:58.951 INFO  [2770241] [CSteppingAction::report@421]  seqmat_mismatch 0
    2016-10-20 11:23:58.951 INFO  [2770241] [CSteppingAction::report@434]  debug_photon 6 (photon_id) 
        5235
        4221
        3186
        2766
        2766
         839
    2016-10-20 11:23:58.951 INFO  [2770241] [CSteppingAction::report@441] TO DEBUG THESE USE:  --dindex=5235,4221,3186,2766,2766,839
    2016-10-20 11:23:58.951 INFO  [2770241] [CG4::postpropagate@296] CG4::postpropagate(0) DONE



pushing out truncation, pushes out the problem 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    tlaser-t --dindex=4124,3285 --bouncemax 15 --recordmax 16 


    2016-10-20 15:27:35.934 INFO  [2830854] [CSteppingAction::report@412]  seqhis_mismatch 2
     rdr cccbcc0ccc9ccccd rec 5ccbcc0ccc9ccccd
     rdr cc6ccccacccccc5d rec 5c6ccccacccccc5d
    2016-10-20 15:27:35.934 INFO  [2830854] [CSteppingAction::report@426]  seqmat_mismatch 0
    2016-10-20 15:27:35.934 INFO  [2830854] [CSteppingAction::report@439]  debug_photon 2 (photon_id) 
        4124
        3285
    2016-10-20 15:27:35.934 INFO  [2830854] [CSteppingAction::report@446] TO DEBUG THESE USE:  --dindex=4124,3285


    tlaser-t --bouncemax 16 --recordmax 16 

    2016-10-20 15:59:31.210 INFO  [2839084] [CSteppingAction::report@412]  seqhis_mismatch 2
     rdr cccacccccc9ccccd rec 5ccacccccc9ccccd
     rdr cccc0b0ccccc6ccd rec 5ccc0b0ccccc6ccd
    2016-10-20 15:59:31.210 INFO  [2839084] [CSteppingAction::report@426]  seqmat_mismatch 0
    2016-10-20 15:59:31.210 INFO  [2839084] [CSteppingAction::report@439]  debug_photon 2 (photon_id) 
        7836
        5501



FIXED : was comparing before all REJOINs are in
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suspect the comparison if happening prior to the
rejoin being completed ... 

Yep. Moved to backwards looking comparison to fix. 



truncation control
~~~~~~~~~~~~~~~~~~~~

::

    409    char bouncemax[128];
    410    snprintf(bouncemax,128,
    411 "Maximum number of boundary bounces, 0:prevents any propagation leaving generated photons"
    412 "Default %d ", m_bouncemax);
    413    m_desc.add_options()
    414        ("bouncemax,b",  boost::program_options::value<int>(&m_bouncemax), bouncemax );
    415 
    416 
    417    // keeping bouncemax one less than recordmax is advantageous 
    418    // as bookeeping is then consistent between the photons and the records 
    419    // as this avoiding truncation of the records
    420 
    421    char recordmax[128];
    422    snprintf(recordmax,128,
    423 "Maximum number of photon step records per photon, 1:to minimize without breaking machinery. Default %d ", m_recordmax);
    424    m_desc.add_options()
    425        ("recordmax,r",  boost::program_options::value<int>(&m_recordmax), recordmax );
    426 




CRecorder m_seqhis 
~~~~~~~~~~~~~~~~~~

primarily from CRecorder::RecordStepPoint based on flag argument and current slot,
note that m_slot continues to increment well past the recording range. 

This means that local *slot* gets will continue to point to m_steps_per_photon - 1 


The mismatch happens prior to lastPost, so problem all from pre::


    488     if(!preSkip)
    489     {
    490        done = RecordStepPoint( pre, preFlag, preMat, m_prior_boundary_status, PRE );
    491     }
    492 
    493     if(lastPost && !done)
    494     {
    495        done = RecordStepPoint( post, postFlag, postMat, m_boundary_status, POST );
    496     }
    497 


Rec m_seqhis
~~~~~~~~~~~~~~~~

Rec::addFlagMaterial attemps to mimmick CRecorder recording based on m_slot and flag argument.
This is invoked based on saved states by Rec::sequence

Hmm the below will always end with POST even prior to lastPost or when truncated... 

::

    298     
    299     for(unsigned i=0 ; i < nstate; i++)
    300     {
    301         rc = getFlagMaterialStageDone(flag, material, stage, done, i, PRE );
    302         if(rc == OK)
    303             addFlagMaterial(flag, material) ;
    304     }
    305     
    306     rc = getFlagMaterialStageDone(flag, material, stage, done, nstate-1, POST );
    307     if(rc == OK)
    308         addFlagMaterial(flag, material) ;




How to proceed ?
------------------

* need to add DYB style reemission to CFG4 

First tack, teleport in the DsG4Scintillation code and try to get it to work::

    simon:cfg4 blyth$ cp /usr/local/env/dyb/NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsG4Scintillation.h .
    simon:cfg4 blyth$ cp /usr/local/env/dyb/NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsG4Scintillation.cc .
    simon:cfg4 blyth$ cp /usr/local/env/dyb/NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsPhysConsOptical.h .



Adopting DYBOp into CFG4
---------------------------

Trying to passalong the primary index in CSteppingAction::setTrack
only works when one reem happens (ie there is at most one call to DsG4Scintillation::PostStepDoIt)
in between steps.  But there are often two such calls.. 

::

    208     if(m_optical)          
    209     {                      
    210          if(m_parent_id == -1) // track is a primary opticalphoton (ie not from reemission)
    211          {                 
    212              G4Track* mtrack = const_cast<G4Track*>(track);
    213 
    214              // m_primary_photon_id++ ;  // <-- starts at -1, thus giving zero-based index
    215              int primary_photon_id = m_track_id ;   // instead of minting new index, use track_id
    216 
    217              mtrack->SetParentID(primary_photon_id);      
    218 
    219              LOG(info) << "CSteppingAction::setTrack"
    220                        << " primary photon "
    221                        << " track_id " << m_track_id
    222                        << " parent_id " << m_parent_id
    223                        << " primary_photon_id " << primary_photon_id 
    224                        ;
    225 
    226          }   
    227          else
    228          {   
    229              LOG(info) << "CSteppingAction::setTrack"
    230                        << " 2ndary photon "
    231                        << " track_id " << m_track_id
    232                        << " parent_id " << m_parent_id << "<-primary" 
    233                        ;
    234          }
    235     }        
    236 }        




::

    2016-10-05 13:02:27.694 INFO  [1902787] [CSteppingAction::setTrack@219] CSteppingAction::setTrack primary photon  track_id 543 parent_id -1 primary_photon_id 543
    2016-10-05 13:02:27.695 INFO  [1902787] [CSteppingAction::setTrack@219] CSteppingAction::setTrack primary photon  track_id 542 parent_id -1 primary_photon_id 542
    2016-10-05 13:02:27.695 INFO  [1902787] [CSteppingAction::setTrack@219] CSteppingAction::setTrack primary photon  track_id 541 parent_id -1 primary_photon_id 541
    2016-10-05 13:02:27.695 INFO  [1902787] [*DsG4Scintillation::PostStepDoIt@771]  DsG4Scintillation reemit  psdi_index 49098 secondaryTime(ns) 2.57509 track_id 540 parent_id -1 scnt 2 nscnt 2
    2016-10-05 13:02:27.695 INFO  [1902787] [CSteppingAction::setTrack@219] CSteppingAction::setTrack primary photon  track_id 540 parent_id -1 primary_photon_id 540
    2016-10-05 13:02:27.695 INFO  [1902787] [*DsG4Scintillation::PostStepDoIt@771]  DsG4Scintillation reemit  psdi_index 49099 secondaryTime(ns) 2.66136 track_id 10440 parent_id 540 scnt 2 nscnt 2
    2016-10-05 13:02:27.695 INFO  [1902787] [CSteppingAction::setTrack@229] CSteppingAction::setTrack 2ndary photon  track_id 10440 parent_id 540<-primary
    2016-10-05 13:02:27.695 WARN  [1902787] [OpPointFlag@266]  reaching...  NoProc
    2016-10-05 13:02:27.695 INFO  [1902787] [CSteppingAction::setTrack@229] CSteppingAction::setTrack 2ndary photon  track_id 10441 parent_id 10440<-primary
    2016-10-05 13:02:27.695 WARN  [1902787] [OpPointFlag@266]  reaching...  NoProc
    2016-10-05 13:02:27.695 INFO  [1902787] [CSteppingAction::setTrack@219] CSteppingAction::setTrack primary photon  track_id 539 parent_id -1 primary_photon_id 539
    2016-10-05 13:02:27.695 INFO  [1902787] [CSteppingAction::setTrack@219] CSteppingAction::setTrack primary photon  track_id 538 parent_id -1 primary_photon_id 538


CRecorder and Rec are almost matching at 10k level : truncation difference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* truncation difference for REJOIN into last slot 

::

    2016-10-05 20:42:04.769 INFO  [2023965] [CSteppingAction::report@383] CG4::postpropagate
     event_total 1
     track_total 10468
     step_total 51335
    2016-10-05 20:42:04.769 INFO  [2023965] [CSteppingAction::report@393]  seqhis_mismatch 6
     rdr       cccc9ccccd rec       5ccc9ccccd
     rdr       cccc9ccccd rec       5ccc9ccccd
     rdr       cccc9ccccd rec       5ccc9ccccd
     rdr       cccc9ccccd rec       5ccc9ccccd
     rdr       cccc9ccccd rec       5ccc9ccccd
     rdr       cccc9ccccd rec       5ccc9ccccd
    2016-10-05 20:42:04.769 INFO  [2023965] [CSteppingAction::report@407]  seqmat_mismatch 0




Hmm seems hijacking ParentID is not so easy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:geant4_10_02_p01 blyth$ find source -name '*.cc' -exec grep -H SetParentID {} \;
    source/error_propagation/src/G4ErrorPropagator.cc:  theG4Track->SetParentID(0);
    source/event/src/G4PrimaryTransformer.cc:    track->SetParentID(0);
    source/event/src/G4StackManager.cc:      aTrack->SetParentID(-1);
    source/processes/electromagnetic/dna/management/src/G4ITModelProcessor.cc:          GetIT(secondary)->SetParentID(trackA->GetTrackID(),
    source/processes/electromagnetic/dna/management/src/G4ITStepProcessor2.cc:    tempSecondaryTrack->SetParentID(fpTrack->GetTrackID());
    source/processes/electromagnetic/dna/utils/src/G4DNAChemistryManager.cc:    H2OTrack -> SetParentID(theIncomingTrack->GetTrackID());
    source/processes/electromagnetic/dna/utils/src/G4DNAChemistryManager.cc:    e_aqTrack -> SetParentID(theIncomingTrack->GetTrackID());
    source/processes/electromagnetic/dna/utils/src/G4DNAChemistryManager.cc:    track -> SetParentID(parentID);
    source/processes/electromagnetic/dna/utils/src/G4DNAChemistryManager.cc:    track -> SetParentID(theIncomingTrack->GetTrackID());
    source/processes/electromagnetic/xrays/src/G4Cerenkov.cc:                aSecondaryTrack->SetParentID(aTrack.GetTrackID());
    source/processes/electromagnetic/xrays/src/G4Scintillation.cc:                aSecondaryTrack->SetParentID(aTrack.GetTrackID());
    source/processes/electromagnetic/xrays/src/G4VXTRenergyLoss.cc:      aSecondaryTrack->SetParentID( aTrack.GetTrackID() );
    source/processes/optical/src/G4OpWLS.cc:    aSecondaryTrack->SetParentID(aTrack.GetTrackID());
    source/tracking/src/G4SteppingManager2.cc:         tempSecondaryTrack->SetParentID( fTrack->GetTrackID() );
    source/tracking/src/G4SteppingManager2.cc:         tempSecondaryTrack->SetParentID( fTrack->GetTrackID() );
    source/tracking/src/G4SteppingManager2.cc:            tempSecondaryTrack->SetParentID( fTrack->GetTrackID() );
    simon:geant4_10_02_p01 blyth$ 


attach primaryPhotonId ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generators create G4PrimaryVertex and add to G4Event::

    255 void CTorchSource::GeneratePrimaryVertex(G4Event *evt)
    256 {
    ...
    275     for (G4int i = 0; i < m_num; i++)
    276     {
    277         pp.position = m_posGen->GenerateOne();
    278         G4PrimaryVertex* vertex = new G4PrimaryVertex(pp.position,m_time);
    ...
    305         G4PrimaryParticle* particle = new G4PrimaryParticle(m_definition);
    ...
    ...
    379         vertex->SetPrimary(particle);
    380         evt->AddPrimaryVertex(vertex);
    ...
    384     }
    385 }


Searching for what happens to G4PrimaryVertex next reveals::

    //  g4-;g4-cls G4PrimaryTransformer

    041 // class description:
     42 //
     43 //  This class is exclusively used by G4EventManager for the conversion
     44 // from G4PrimaryVertex/G4PrimaryParticle to G4DynamicParticle/G4Track.
     45 //
     46 
     47 class G4PrimaryTransformer
     48 {

    115 void G4PrimaryTransformer::GenerateSingleTrack
    116      (G4PrimaryParticle* primaryParticle,
    117       G4double x0,G4double y0,G4double z0,G4double t0,G4double wv)
    118 {
    ...
    ...
    218     // Create G4Track object
    219     G4Track* track = new G4Track(DP,t0,G4ThreeVector(x0,y0,z0));
    220     // Set trackID and let primary particle know it
    221     trackID++;
    222     track->SetTrackID(trackID);
    223     primaryParticle->SetTrackID(trackID);
    224     // Set parentID to 0 as a primary particle
    225     track->SetParentID(0);
    226     // Set weight ( vertex weight * particle weight )
    227     track->SetWeight(wv*(primaryParticle->GetWeight()));
    228     // Store it to G4TrackVector
    229     TV.push_back( track );
    230 
    231   }
    232 }






flags borked, so flying blind
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* lots of Undefined boundary status


tlaser-;tlaser-d;tlaser.py::

      A:seqhis_ana      1:laser 
              8ccccd        0.767           7673       [6 ] TO BT BT BT BT SA
                  4d        0.055            553       [2 ] TO AB
          cccc9ccccd        0.024            242       [10] TO BT BT BT BT DR BT BT BT BT
             8cccc6d        0.019            188       [7 ] TO SC BT BT BT BT SA
                4ccd        0.012            122       [4 ] TO BT BT AB
             8cccc5d        0.012            121       [7 ] TO RE BT BT BT BT SA
                 45d        0.006             65       [3 ] TO RE AB
              4ccccd        0.006             63       [6 ] TO BT BT BT BT AB
            8cccc55d        0.005             52       [8 ] TO RE RE BT BT BT BT SA
             8cc6ccd        0.004             39       [7 ] TO BT BT SC BT BT SA
                455d        0.003             34       [4 ] TO RE RE AB
          cccccc6ccd        0.003             34       [10] TO BT BT SC BT BT BT BT BT BT
             8cc5ccd        0.003             27       [7 ] TO BT BT RE BT BT SA
             86ccccd        0.003             27       [7 ] TO BT BT BT BT SC SA
           8cccc555d        0.003             26       [9 ] TO RE RE RE BT BT BT BT SA
               4cccd        0.003             25       [5 ] TO BT BT BT AB
          cacccccc5d        0.002             22       [10] TO RE BT BT BT BT BT BT SR BT
                 46d        0.002             21       [3 ] TO SC AB
          cccc6ccccd        0.002             20       [10] TO BT BT BT BT SC BT BT BT BT
            4ccccc5d        0.002             19       [8 ] TO RE BT BT BT BT BT AB
                           10000         1.00 
       B:seqhis_ana     -1:laser 
                   0        0.850           8498       [1 ] ?0?
                  4d        0.071            708       [2 ] TO AB
                   d        0.028            276       [1 ] TO
                400d        0.017            168       [4 ] TO ?0? ?0? AB
              40000d        0.009             92       [6 ] TO ?0? ?0? ?0? ?0? AB
                  6d        0.008             82       [2 ] TO SC
                600d        0.004             35       [4 ] TO ?0? ?0? SC
                 46d        0.003             26       [3 ] TO SC AB
              60000d        0.002             16       [6 ] TO ?0? ?0? ?0? ?0? SC
               4000d        0.002             15       [5 ] TO ?0? ?0? ?0? AB
          400000000d        0.002             15       [10] TO ?0? ?0? ?0? ?0? ?0? ?0? ?0? ?0? AB
                 40d        0.001             11       [3 ] TO ?0? AB
            4000000d        0.001              7       [8 ] TO ?0? ?0? ?0? ?0? ?0? ?0? AB
             400600d        0.001              6       [7 ] TO ?0? ?0? SC ?0? ?0? AB
               4006d        0.001              6       [5 ] TO SC ?0? ?0? AB
          600000000d        0.001              6       [10] TO ?0? ?0? ?0? ?0? ?0? ?0? ?0? ?0? SC
             400006d        0.000              4       [7 ] TO SC ?0? ?0? ?0? ?0? AB
                 66d        0.000              3       [3 ] TO SC SC
               6006d        0.000              3       [5 ] TO SC ?0? ?0? SC
               6000d        0.000              3       [5 ] TO ?0? ?0? ?0? SC
                           10000         1.00 

Regained flags with USE_CUSTOM_BOUNDARY flipping::

      A:seqhis_ana      1:laser 
              8ccccd        0.767           7673       [6 ] TO BT BT BT BT SA
                  4d        0.055            553       [2 ] TO AB
          cccc9ccccd        0.024            242       [10] TO BT BT BT BT DR BT BT BT BT
             8cccc6d        0.019            188       [7 ] TO SC BT BT BT BT SA
                4ccd        0.012            122       [4 ] TO BT BT AB
             8cccc5d        0.012            121       [7 ] TO RE BT BT BT BT SA
                 45d        0.006             65       [3 ] TO RE AB
              4ccccd        0.006             63       [6 ] TO BT BT BT BT AB
            8cccc55d        0.005             52       [8 ] TO RE RE BT BT BT BT SA
             8cc6ccd        0.004             39       [7 ] TO BT BT SC BT BT SA
                455d        0.003             34       [4 ] TO RE RE AB
          cccccc6ccd        0.003             34       [10] TO BT BT SC BT BT BT BT BT BT
             8cc5ccd        0.003             27       [7 ] TO BT BT RE BT BT SA
             86ccccd        0.003             27       [7 ] TO BT BT BT BT SC SA
           8cccc555d        0.003             26       [9 ] TO RE RE RE BT BT BT BT SA
               4cccd        0.003             25       [5 ] TO BT BT BT AB
          cacccccc5d        0.002             22       [10] TO RE BT BT BT BT BT BT SR BT
                 46d        0.002             21       [3 ] TO SC AB
          cccc6ccccd        0.002             20       [10] TO BT BT BT BT SC BT BT BT BT
            4ccccc5d        0.002             19       [8 ] TO RE BT BT BT BT BT AB
                           10000         1.00 
       B:seqhis_ana     -1:laser 
              8ccccd        0.811           8110       [6 ] TO BT BT BT BT SA
                  4d        0.075            750       [2 ] TO AB
          cccc9ccccd        0.024            238       [10] TO BT BT BT BT DR BT BT BT BT
             8cccc6d        0.018            177       [7 ] TO SC BT BT BT BT SA
                4ccd        0.016            161       [4 ] TO BT BT AB
              4ccccd        0.010            101       [6 ] TO BT BT BT BT AB
             8cc6ccd        0.004             44       [7 ] TO BT BT SC BT BT SA
             86ccccd        0.003             27       [7 ] TO BT BT BT BT SC SA
             89ccccd        0.003             27       [7 ] TO BT BT BT BT DR SA
                 46d        0.003             26       [3 ] TO SC AB
               4cccd        0.002             22       [5 ] TO BT BT BT AB
          cacccccc6d        0.002             22       [10] TO SC BT BT BT BT BT BT SR BT
            8ccccc6d        0.002             21       [8 ] TO SC BT BT BT BT BT SA
          cccccc6ccd        0.002             20       [10] TO BT BT SC BT BT BT BT BT BT
          cccc6ccccd        0.002             16       [10] TO BT BT BT BT SC BT BT BT BT
          ccbccccc6d        0.002             15       [10] TO SC BT BT BT BT BT BR BT BT
           4cc9ccccd        0.001             14       [9 ] TO BT BT BT BT DR BT BT AB
           cac0ccc6d        0.001             14       [9 ] TO SC BT BT BT ?0? BT SR BT
                 4cd        0.001             13       [3 ] TO BT AB
             49ccccd        0.001              9       [7 ] TO BT BT BT BT DR AB
                           10000         1.00 





live reemission photon counts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

STATIC buffer was expecting a certain number of photons, so currently truncates::

    2016-10-04 11:49:41.787 INFO  [1669872] [CSteppingAction::UserSteppingAction@156] CSA (startEvent) event_id 9 event_total 9
    2016-10-04 11:49:41.787 INFO  [1669872] [CSteppingAction::UserSteppingActionOptical@320] CSA::UserSteppingActionOptical NOT RECORDING  record_id 100000 record_max 100000 STATIC 
    2016-10-04 11:49:41.787 INFO  [1669872] [CSteppingAction::UserSteppingActionOptical@320] CSA::UserSteppingActionOptical NOT RECORDING  record_id 100000 record_max 100000 STATIC 
    ...
    2016-10-04 11:49:42.529 INFO  [1669872] [CSteppingAction::UserSteppingActionOptical@320] CSA::UserSteppingActionOptical NOT RECORDING  record_id 100495 record_max 100000 STATIC 
    2016-10-04 11:49:42.529 INFO  [1669872] [CSteppingAction::UserSteppingActionOptical@320] CSA::UserSteppingActionOptical NOT RECORDING  record_id 100495 record_max 100000 STATIC 
    2016-10-04 11:49:42.532 INFO  [1669872] [CRunAction::EndOfRunAction@23] CRunAction::EndOfRunAction count 1


Hmm, I wonder if all the "NOT RECORDING" are RE ?  Looks to be so


Normally with fabricated (as opposed to G4 live) gensteps, the number of photons is known ahead of time.

Reemission means cannot know photon counts ahead of time ?

* that statement is true only if you count reemits as new photons, Opticks does not do that
 
Contining the slot for reemiisions with G4 ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is necessary for easy comparisons between G4 and Opticks.

With Opticks a reemitted photon continues the lineage (buffer slot) 
of its predecessor but with G4 a fresh new particle is created ...  

Small scale less than 10k photon torch running (corresponding to a single G4 "subevt") 
looks like can effect a continuation of reemission photons using the parent_id.  

For over 10k, need to cope with finding parent "subevt" too to line up with the correct 
record number. Unless can be sure subevt dont handled in mixed order ?

::

    2016-10-04 15:01:45.104 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] S-R photon_id     219 parent_id      -1 step_id    0 record_id     219 record_max   10000 STATIC 
    2016-10-04 15:01:45.104 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] S-R photon_id     218 parent_id      -1 step_id    0 record_id     218 record_max   10000 STATIC 
    2016-10-04 15:01:45.104 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] S-R photon_id     217 parent_id      -1 step_id    0 record_id     217 record_max   10000 STATIC 
    2016-10-04 15:01:45.104 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] S-R photon_id     216 parent_id      -1 step_id    0 record_id     216 record_max   10000 STATIC 
    2016-10-04 15:01:45.104 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] S-R photon_id     215 parent_id      -1 step_id    0 record_id     215 record_max   10000 STATIC 
    2016-10-04 15:01:45.104 INFO  [1721635] [*DsG4Scintillation::PostStepDoIt@761] reemit secondaryTime(ns) 18.6468 parent_id 215
    2016-10-04 15:01:45.104 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] SC- photon_id   10454 parent_id     215 step_id    0 record_id   10454 record_max   10000 STATIC 
    2016-10-04 15:01:45.104 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] -C- photon_id   10454 parent_id     215 step_id    1 record_id   10454 record_max   10000 STATIC 
    2016-10-04 15:01:45.104 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] -C- photon_id   10454 parent_id     215 step_id    2 record_id   10454 record_max   10000 STATIC 
    2016-10-04 15:01:45.104 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] S-R photon_id     214 parent_id      -1 step_id    0 record_id     214 record_max   10000 STATIC 
    2016-10-04 15:01:45.104 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] S-R photon_id     213 parent_id      -1 step_id    0 record_id     213 record_max   10000 STATIC 
    2016-10-04 15:01:45.104 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] S-R photon_id     212 parent_id      -1 step_id    0 record_id     212 record_max   10000 STATIC 
    2016-10-04 15:01:45.104 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] S-R photon_id     211 parent_id      -1 step_id    0 record_id     211 record_max   10000 STATIC 
    2016-10-04 15:01:45.105 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] S-R photon_id     210 parent_id      -1 step_id    0 record_id     210 record_max   10000 STATIC 
    2016-10-04 15:01:45.105 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] S-R photon_id     209 parent_id      -1 step_id    0 record_id     209 record_max   10000 STATIC 
    2016-10-04 15:01:45.105 INFO  [1721635] [CSteppingAction::UserSteppingActionOptical@291] S-R photon_id     208 parent_id      -1 step_id    0 record_id     208 record_max   10000 STATIC 


will the reemit step always come immediately after its parent one...  note the reversed photon order
what about multiple reemissions 

otherwise need to record the slots for all photons in order to continue them ?

::

    2016-10-04 18:12:58.303 INFO  [1777349] [CSteppingAction::UserSteppingActionOptical@296] S-R photon_id      21 parent_id      -1 step_id    0 record_id      21 record_max      50 event_id       0 pre     0.1 post 8.05857 STATIC 
    2016-10-04 18:12:58.303 INFO  [1777349] [CSteppingAction::UserSteppingActionOptical@296] S-R photon_id      20 parent_id      -1 step_id    0 record_id      20 record_max      50 event_id       0 pre     0.1 post 8.05857 STATIC 
    2016-10-04 18:12:58.303 INFO  [1777349] [CSteppingAction::UserSteppingActionOptical@296] S-R photon_id      19 parent_id      -1 step_id    0 record_id      19 record_max      50 event_id       0 pre     0.1 post 8.05857 STATIC 
    2016-10-04 18:12:58.303 INFO  [1777349] [CSteppingAction::UserSteppingActionOptical@296] S-R photon_id      18 parent_id      -1 step_id    0 record_id      18 record_max      50 event_id       0 pre     0.1 post 8.05857 STATIC 
    2016-10-04 18:12:58.303 INFO  [1777349] [*DsG4Scintillation::PostStepDoIt@761] reemit secondaryTime(ns) 1.48211 parent_id 17
    2016-10-04 18:12:58.303 INFO  [1777349] [CSteppingAction::UserSteppingActionOptical@296] S-R photon_id      17 parent_id      -1 step_id    0 record_id      17 record_max      50 event_id       0 pre     0.1 post 1.48211 STATIC 
    2016-10-04 18:12:58.303 INFO  [1777349] [CSteppingAction::UserSteppingActionOptical@296] SC- photon_id      50 parent_id      17 step_id    0 record_id      50 record_max      50 event_id       0 pre 1.48211 post 6.09097 STATIC 
    2016-10-04 18:12:58.303 INFO  [1777349] [CSteppingAction::UserSteppingActionOptical@296] S-R photon_id      16 parent_id      -1 step_id    0 record_id      16 record_max      50 event_id       0 pre     0.1 post 8.05857 STATIC 
    2016-10-04 18:12:58.303 INFO  [1777349] [CSteppingAction::UserSteppingActionOptical@296] S-R photon_id      15 parent_id      -1 step_id    0 record_id      15 record_max      50 event_id       0 pre     0.1 post 0.489073 STATIC 
    2016-10-04 18:12:58.303 INFO  [1777349] [CSteppingAction::UserSteppingActionOptical@296] S-R photon_id      14 parent_id      -1 step_id    0 record_id      14 record_max      50 event_id       0 pre     0.1 post 8.05857 STATIC 



reemission continuation are difficult to implement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

G4 produces secondary reemission photon with large trackId, which then have task of
linking with the fixed set of photons, within the recording range. 

When the parent id of the 2ndary photon matches the last_photon_id 
is a simple RHOP and can just continue filling slots.

Similarly when grandparent id photon matches last_photon_id can
just continue.

::

    318     int last_photon_id = m_recorder->getPhotonId();
    319 
    320     RecStage_t stage = UNKNOWN ;
    321     if( parent_id == -1 )
    322     {
    323         stage = photon_id != last_photon_id  ? START : COLLECT ;
    324     }
    325     else if( parent_id >= 0 && parent_id == last_photon_id )
    326     {
    327         stage = RHOP ;
    328         photon_id = parent_id ;
    329     }
    330     else if( grandparent_id >= 0 && grandparent_id == last_photon_id )
    331     {
    332         stage = RJUMP ;
    333         photon_id = grandparent_id ;
    334     }
    335 
    336 
    337     m_recorder->setPhotonId(photon_id);
    338     m_recorder->setEventId(eid);
    339     m_recorder->setStepId(step_id);
    340     m_recorder->setParentId(parent_id);




* difficult to make the connection between the secondary and the parent/grandparent
  that the new photons are in lineage with

* how can avoid the AB ? and getting stuck in 


::


     A:seqhis_ana      1:laser 
              8ccccd        0.756            756       [6 ] TO BT BT BT BT SA
                  4d        0.063             63       [2 ] TO AB
          cccc9ccccd        0.026             26       [10] TO BT BT BT BT DR BT BT BT BT
             8cccc6d        0.021             21       [7 ] TO SC BT BT BT BT SA
             8cccc5d        0.012             12       [7 ] TO RE BT BT BT BT SA
                4ccd        0.011             11       [4 ] TO BT BT AB
              4ccccd        0.007              7       [6 ] TO BT BT BT BT AB
                 45d        0.005              5       [3 ] TO RE AB
           8cccc555d        0.005              5       [9 ] TO RE RE RE BT BT BT BT SA
             8cc6ccd        0.005              5       [7 ] TO BT BT SC BT BT SA
            4ccccc5d        0.005              5       [8 ] TO RE BT BT BT BT BT AB
            8cccc55d        0.005              5       [8 ] TO RE RE BT BT BT BT SA
                 4cd        0.003              3       [3 ] TO BT AB
                455d        0.003              3       [4 ] TO RE RE AB
             86ccccd        0.003              3       [7 ] TO BT BT BT BT SC SA
            4ccccc6d        0.003              3       [8 ] TO SC BT BT BT BT BT AB
            8cc55ccd        0.003              3       [8 ] TO BT BT RE RE BT BT SA
          cccccc6ccd        0.003              3       [10] TO BT BT SC BT BT BT BT BT BT
          cccc55555d        0.003              3       [10] TO RE RE RE RE RE BT BT BT BT
          ccc9cccc6d        0.002              2       [10] TO SC BT BT BT BT DR BT BT BT
                            1000         1.00 
       B:seqhis_ana     -1:laser 
              8ccccd        0.817            817       [6 ] TO BT BT BT BT SA
                  4d        0.060             60       [2 ] TO AB
          cccc9ccccd        0.024             24       [10] TO BT BT BT BT DR BT BT BT BT
             8cccc6d        0.009              9       [7 ] TO SC BT BT BT BT SA
                4ccd        0.007              7       [4 ] TO BT BT AB
              45454d        0.005              5       [6 ] TO AB RE AB RE AB   
              4ccccd        0.005              5       [6 ] TO BT BT BT BT AB
          cccccc6ccd        0.005              5       [10] TO BT BT SC BT BT BT BT BT BT
            8ccccc6d        0.003              3       [8 ] TO SC BT BT BT BT BT SA
            8cccc54d        0.003              3       [8 ] TO AB RE BT BT BT BT SA
           ccc9ccccd        0.003              3       [9 ] TO BT BT BT BT DR BT BT BT
          8cccc5454d        0.003              3       [10] TO AB RE AB RE BT BT BT BT SA
               4cccd        0.003              3       [5 ] TO BT BT BT AB
                 46d        0.003              3       [3 ] TO SC AB
             86ccccd        0.003              3       [7 ] TO BT BT BT BT SC SA
             8cc6ccd        0.003              3       [7 ] TO BT BT SC BT BT SA
           8cccc654d        0.002              2       [9 ] TO AB RE SC BT BT BT BT SA
          8cbccccc6d        0.002              2       [10] TO SC BT BT BT BT BT BR BT SA
             8ccc6cd        0.002              2       [7 ] TO BT SC BT BT BT SA
          cacccccc6d        0.002              2       [10] TO SC BT BT BT BT BT BT SR BT
                            1000         1.00 


Must less RE in CG4 ? Scrubbing the AB by going back one slot and replace with RE::

       A:seqhis_ana      1:laser 
              8ccccd        0.764         763501       [6 ] TO BT BT BT BT SA
                  4d        0.056          55825       [2 ] TO AB
          cccc9ccccd        0.025          25263       [10] TO BT BT BT BT DR BT BT BT BT
             8cccc6d        0.020          19707       [7 ] TO SC BT BT BT BT SA
                4ccd        0.013          12576       [4 ] TO BT BT AB
             8cccc5d        0.011          11183       [7 ] TO RE BT BT BT BT SA
              4ccccd        0.009           8554       [6 ] TO BT BT BT BT AB
                 45d        0.008           7531       [3 ] TO RE AB
            8cccc55d        0.005           5362       [8 ] TO RE RE BT BT BT BT SA
             8cc6ccd        0.004           4109       [7 ] TO BT BT SC BT BT SA
                455d        0.004           3588       [4 ] TO RE RE AB
             86ccccd        0.003           2836       [7 ] TO BT BT BT BT SC SA
          cccccc6ccd        0.003           2674       [10] TO BT BT SC BT BT BT BT BT BT
           8cccc555d        0.003           2524       [9 ] TO RE RE RE BT BT BT BT SA
             8cc5ccd        0.002           2359       [7 ] TO BT BT RE BT BT SA
          cacccccc6d        0.002           2210       [10] TO SC BT BT BT BT BT BT SR BT
                 46d        0.002           2118       [3 ] TO SC AB
          cccc6ccccd        0.002           2060       [10] TO BT BT BT BT SC BT BT BT BT
               4cccd        0.002           1940       [5 ] TO BT BT BT AB
             89ccccd        0.002           1880       [7 ] TO BT BT BT BT DR SA
                         1000000         1.00 
       B:seqhis_ana     -1:laser 
              8ccccd        0.814         813976       [6 ] TO BT BT BT BT SA
                  4d        0.048          48056       [2 ] TO AB
          cccc9ccccd        0.026          26149       [10] TO BT BT BT BT DR BT BT BT BT
             8cccc6d        0.019          18604       [7 ] TO SC BT BT BT BT SA
                4ccd        0.012          11614       [4 ] TO BT BT AB
                 8cd        0.010          10193       [3 ] TO BT SA
              4ccccd        0.009           8755       [6 ] TO BT BT BT BT AB
             8cc6ccd        0.004           4157       [7 ] TO BT BT SC BT BT SA
                  8d        0.004           3614       [2 ] TO SA
               8cccd        0.003           2746       [5 ] TO BT BT BT SA
             86ccccd        0.003           2696       [7 ] TO BT BT BT BT SC SA
                8c5d        0.002           2454       [4 ] TO RE BT SA
                455d        0.002           2354       [4 ] TO RE RE AB
                 45d        0.002           2306       [3 ] TO RE AB
               4cccd        0.002           2244       [5 ] TO BT BT BT AB
             89ccccd        0.002           2241       [7 ] TO BT BT BT BT DR SA
          cacccccc6d        0.002           2172       [10] TO SC BT BT BT BT BT BT SR BT
                 4cd        0.002           1967       [3 ] TO BT AB
          cccccc6ccd        0.002           1931       [10] TO BT BT SC BT BT BT BT BT BT
            8ccccc6d        0.002           1787       [8 ] TO SC BT BT BT BT BT SA
                         1000000         1.00 



REEMISSIONPROB is not a standard G4 property
----------------------------------------------

::

       +X horizontal tlaser from middle of DYB AD

       A: opticks, has reemission treatment aiming to match DYB NuWa DetSim 
                   (it is handled as a subset of BULK_ABSORB that confers rebirth)

       B: almost stock Geant4 10.2, no reemission treatment -> hence more absorption
                   (stock G4 is just absorbing, and the REEMISSIONPROB is ignored)


       A:seqhis_ana      1:laser 
              8ccccd        0.764         763501       [6 ] TO BT BT BT BT SA
                  4d        0.056          55825       [2 ] TO AB
          cccc9ccccd        0.025          25263       [10] TO BT BT BT BT DR BT BT BT BT
             8cccc6d        0.020          19707       [7 ] TO SC BT BT BT BT SA
                4ccd        0.013          12576       [4 ] TO BT BT AB
             8cccc5d        0.011          11183       [7 ] TO RE BT BT BT BT SA
              4ccccd        0.009           8554       [6 ] TO BT BT BT BT AB
                 45d        0.008           7531       [3 ] TO RE AB
            8cccc55d        0.005           5362       [8 ] TO RE RE BT BT BT BT SA
             8cc6ccd        0.004           4109       [7 ] TO BT BT SC BT BT SA
                455d        0.004           3588       [4 ] TO RE RE AB
             86ccccd        0.003           2836       [7 ] TO BT BT BT BT SC SA
          cccccc6ccd        0.003           2674       [10] TO BT BT SC BT BT BT BT BT BT
           8cccc555d        0.003           2524       [9 ] TO RE RE RE BT BT BT BT SA
             8cc5ccd        0.002           2359       [7 ] TO BT BT RE BT BT SA
          cacccccc6d        0.002           2210       [10] TO SC BT BT BT BT BT BT SR BT
                 46d        0.002           2118       [3 ] TO SC AB
          cccc6ccccd        0.002           2060       [10] TO BT BT BT BT SC BT BT BT BT
               4cccd        0.002           1940       [5 ] TO BT BT BT AB
             89ccccd        0.002           1880       [7 ] TO BT BT BT BT DR SA
                         1000000         1.00 
       B:seqhis_ana     -1:laser 
              8ccccd        0.813         813472       [6 ] TO BT BT BT BT SA
                  4d        0.072          71523       [2 ] TO AB
          cccc9ccccd        0.027          27170       [10] TO BT BT BT BT DR BT BT BT BT
                4ccd        0.017          17386       [4 ] TO BT BT AB
             8cccc6d        0.015          15107       [7 ] TO SC BT BT BT BT SA
              4ccccd        0.009           8842       [6 ] TO BT BT BT BT AB
          cacccccc6d        0.004           3577       [10] TO SC BT BT BT BT BT BT SR BT
             8cc6ccd        0.003           3466       [7 ] TO BT BT SC BT BT SA
                 46d        0.003           2515       [3 ] TO SC AB
             86ccccd        0.002           2476       [7 ] TO BT BT BT BT SC SA
           cac0ccc6d        0.002           2356       [9 ] TO SC BT BT BT ?0? BT SR BT
          cccccc6ccd        0.002           2157       [10] TO BT BT SC BT BT BT BT BT BT
             89ccccd        0.002           2127       [7 ] TO BT BT BT BT DR SA
               4cccd        0.002           1977       [5 ] TO BT BT BT AB
          cccc6ccccd        0.002           1949       [10] TO BT BT BT BT SC BT BT BT BT
            8ccccc6d        0.002           1515       [8 ] TO SC BT BT BT BT BT SA
          ccbccccc6d        0.001           1429       [10] TO SC BT BT BT BT BT BR BT BT
           4cc9ccccd        0.001           1215       [9 ] TO BT BT BT BT DR BT BT AB
                 4cd        0.001           1077       [3 ] TO BT AB
               4cc6d        0.001            802       [5 ] TO SC BT BT AB
                         1000000         1.00 



/usr/local/env/dyb/NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsG4Scintillation.h::

    /// NB unlike stock G4  DsG4Scintillation::IsApplicable is true for opticalphoton
    ///    --> this is needed in order to handle the reemission of optical photons

    300 inline
    301 G4bool DsG4Scintillation::IsApplicable(const G4ParticleDefinition& aParticleType)
    302 {
    303         if (aParticleType.GetParticleName() == "opticalphoton"){
    304            return true;
    305         } else {
    306            return true;
    307         }
    308 }

    ///    NB the untrue comment, presumably inherited from stock G4 
    ///
    137         G4bool IsApplicable(const G4ParticleDefinition& aParticleType);
    138         // Returns true -> 'is applicable', for any particle type except
    139         // for an 'opticalphoton' 



/usr/local/env/dyb/NuWa-trunk/dybgaudi/Simulation/DetSim/src/DsG4Scintillation.cc::

    099 DsG4Scintillation::DsG4Scintillation(const G4String& processName,
    100                                      G4ProcessType type)
    101     : G4VRestDiscreteProcess(processName, type)
    102     , doReemission(true)
    103     , doBothProcess(true)
    104     , fPhotonWeight(1.0)
    105     , fApplyPreQE(false)
    106     , fPreQE(1.)
    107     , m_noop(false)
    108 {
    109     SetProcessSubType(fScintillation);
    110     fTrackSecondariesFirst = false;



    170 G4VParticleChange*
    171 DsG4Scintillation::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
    172 
    173 // This routine is called for each tracking step of a charged particle
    174 // in a scintillator. A Poisson/Gauss-distributed number of photons is 
    175 // generated according to the scintillation yield formula, distributed 
    176 // evenly along the track segment and uniformly into 4pi.
    177 
    178 {
    179     aParticleChange.Initialize(aTrack);
    ...
    187     G4String pname="";
    188     G4ThreeVector vertpos;
    189     G4double vertenergy=0.0;
    190     G4double reem_d=0.0;
    191     G4bool flagReemission= false;

    193     if (aTrack.GetDefinition() == G4OpticalPhoton::OpticalPhoton()) 
            {
    194         G4Track *track=aStep.GetTrack();
    197 
    198         const G4VProcess* process = track->GetCreatorProcess();
    199         if(process) pname = process->GetProcessName();

    ///         flagReemission is set only for opticalphotons that are 
    ///         about to be bulk absorbed(fStopAndKill and !fGeomBoundary)
    ///
    ///           doBothProcess = true :  reemission for optical photons generated by both scintillation and Cerenkov processes         
    ///           doBothProcess = false : reemission for optical photons generated by Cerenkov process only 
    ///

    200 
    204         if(doBothProcess) 
               {
    205             flagReemission= doReemission
    206                 && aTrack.GetTrackStatus() == fStopAndKill
    207                 && aStep.GetPostStepPoint()->GetStepStatus() != fGeomBoundary;
    208         }
    209         else
                {
    210             flagReemission= doReemission
    211                 && aTrack.GetTrackStatus() == fStopAndKill
    212                 && aStep.GetPostStepPoint()->GetStepStatus() != fGeomBoundary
    213                 && pname=="Cerenkov";
    214         }
    218         if (!flagReemission) 
                {
    ///          -> give up the ghost and get absorbed
    219              return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    220         }
    221     }
    223     G4double TotalEnergyDeposit = aStep.GetTotalEnergyDeposit();
    228     if (TotalEnergyDeposit <= 0.0 && !flagReemission) {
    229         return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    230     }
    ...
    246     if (aParticleName == "opticalphoton") {
    247       FastTimeConstant = "ReemissionFASTTIMECONSTANT";
    248       SlowTimeConstant = "ReemissionSLOWTIMECONSTANT";
    249       strYieldRatio = "ReemissionYIELDRATIO";
    250     }
    251     else if(aParticleName == "gamma" || aParticleName == "e+" || aParticleName == "e-") {
    252       FastTimeConstant = "GammaFASTTIMECONSTANT";
    ...
            }

    273     const G4MaterialPropertyVector* Fast_Intensity  = aMaterialPropertiesTable->GetProperty("FASTCOMPONENT");
    275     const G4MaterialPropertyVector* Slow_Intensity  = aMaterialPropertiesTable->GetProperty("SLOWCOMPONENT");
    277     const G4MaterialPropertyVector* Reemission_Prob = aMaterialPropertiesTable->GetProperty("REEMISSIONPROB");
    ...
    283     if (!Fast_Intensity && !Slow_Intensity )
    284         return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    ...
    286     G4int nscnt = 1;
    287     if (Fast_Intensity && Slow_Intensity) nscnt = 2;
    ...
    291     G4StepPoint* pPreStepPoint  = aStep.GetPreStepPoint();
    292     G4StepPoint* pPostStepPoint = aStep.GetPostStepPoint();
    293 
    294     G4ThreeVector x0 = pPreStepPoint->GetPosition();
    295     G4ThreeVector p0 = aStep.GetDeltaPosition().unit();
    296     G4double      t0 = pPreStepPoint->GetGlobalTime();
    297 
    298     //Replace NumPhotons by NumTracks
    299     G4int NumTracks=0;
    300     G4double weight=1.0;
    301     if (flagReemission) 
            {
    ...
    305         if ( Reemission_Prob == 0) return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    307         G4double p_reemission= Reemission_Prob->GetProperty(aTrack.GetKineticEnergy());
    309         if (G4UniformRand() >= p_reemission) return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
    ////
    ////        above line reemission has a chance to not happen, otherwise we create a single secondary...
    ///         conferring reemission "rebirth"
    ////

    311         NumTracks= 1;
    312         weight= aTrack.GetWeight();
    316     else {
    317         //////////////////////////////////// Birks' law ////////////////////////





