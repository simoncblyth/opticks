ts19-g4-extra-point-migrations
===============================

Investigate two maligned history categories  

* (7/1M) photons bouncing on ellipsoid/cone edge
* (105/1M) photons skirting along base tubs edge 


thoughts
---------

* testing with photons neatly aligned, normal incidences etc.. is far more likely to cause edge skimmers that real operation  
* current geometry motivated testing is bound to find such issues 
* allowing photons to bounce around inside PMT is prone to meeting edges, whereas is reality 
  will SA/SD at cathode 

* better use more physically motivated testing with photons coming from expected directions on PMT assembly 
  with cathode collection


DONE : added migtab to ana/ab.py
------------------------------------

* have accounted for (105+7)/166 maligned : categorize all the migrations 
* automate creation of migration tables from the maligned, to see the counts 

::

    from collections import OrderedDict as odict
    d = odict()
    for i,m in enumerate(mal):
        k = (a.seqhis[m],b.seqhis[m])
        if not k in d: d[k]=0 
        d[k] += 1  
        print( " %4d  %16x : %16x " % (i, a.seqhis[m], b.seqhis[m] ))

::

    In [23]: for kv in sorted(d.items(), key=lambda kv:kv[1], reverse=True): print(" %4d   %16x  %16x " % ( kv[1], kv[0][0], kv[0][1] ) ) 
      105              8cccd            8ccccd 
       16              8cccd           8cbcccd 
        7             8cbbcd           8cbbbcd 
        6                8cd              8ccd 
        4              8cccd              8ccd 
        4            8cbbccd             8bccd 
        3              8cccd             8bccd 
        3            8cbbccd            8ccccd 
        3              8cbcd           8cbbbcd 
        2              8cbcd        bbbbbbbbcd 
        2                8bd              8ccd 
        2               8ccd             8cbcd 
        1            8b6cbcd            86cbcd 
        1                8bd             8cbcd 
        1               8bcd             8cbcd 
        1            8ccbbcd          8cccbbcd 
        1            8ccbbcd           8bcbbcd 
        1             86cbcd          8cc6cbcd 
        1             8cbccd              8ccd 
        1            8bbbbcd          8cbbbbcd 
        1             8cbbcd        bbbbbbbbcd 

    In [30]: s = 0 

    In [31]: for kv in sorted(d.items(), key=lambda kv:kv[1], reverse=True): s += kv[1]

    In [32]: s
    Out[32]: 166



objective : investigate some migrations
-------------------------------------------
::

   TAG=3 ta 19 --pfx scan-ts-utail --msli :1M

::

    ab.mal
    aligned   999834/1000000 : 0.9998 : 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 
    maligned     166/1000000 : 0.0002 : 2908,4860,5477,12338,17891,18117,28709,32764,37671,43675,45874,46032,60178,63381,72351,76458,78372,86277,95271,99872,114621,114824,117993,124178,128075 
    slice(0, 25, None)
          0   2908 : * :                                  TO BT BR BR BT SA                               TO BT BR BR BR BT SA 
          1   4860 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
          2   5477 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 




shoot from the top, still gives substantial maligned as happens across other side
---------------------------------------------------------------------------------------

* changed tboolean-proxy sheetmask to 0x2 to shoot from +Z

::

    TAG=5 ts 19 --pfx scan-ts-utail 


Gives 8/10k maligned L mostly extra BR from G4.::

    ab.mal
    aligned     9992/  10000 : 0.9992 : 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 
    maligned       8/  10000 : 0.0008 : 938,944,1016,4007,5449,5985,8006,8473 
    slice(0, 25, None)
          0    938 : * :                                        TO BT BT SA                                     TO BT BR BT SA 
          1    944 : * :                                  TO BT BR BR BT SA                                     TO BT BR BT SA 
          2   1016 : * :                                     TO BT BR BR SA                                     TO BT BR BT SA 
          3   4007 : * :                                        TO BT BT SA                                     TO BT BR BT SA 
          4   5449 : * :                                        TO BT BT SA                                     TO BT BR BT SA 
          5   5985 : * :                                     TO BT BR BR SA                                     TO BT BR BT SA 
          6   8006 : * :                                        TO BT BT SA                                     TO BT BR BT SA 
          7   8473 : * :                                        TO BT BT SA                                     TO BT BR BT SA 

Viewing just those they make beeline for edges::

    TAG=5 ts 19 --pfx scan-ts-utail --mask=938,944,1016,4007,5449,5985,8006,8473


Up the stats, gives 799/1M maligned::

    TAG=5 ts 19 --pfx scan-ts-utail --generateoverride -1 
    TAG=5 ta 19 --pfx scan-ts-utail --msli :1M

::

    b.mal
    aligned   999201/1000000 : 0.9992 : 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 
    maligned     799/1000000 : 0.0008 : 938,944,1016,4007,5449,5985,8006,8473,11571,14233,14618,14633,15370,15417,16913,19782,21866,22269,22415,24391,24600,25702,26823,27536,29423 


    ab.mal.migtab
      551               8ccd                    TO BT BT SA              8cbcd                 TO BT BR BT SA  
      161              8bbcd                 TO BT BR BR SA              8cbcd                 TO BT BR BT SA  
       33               8ccd                    TO BT BT SA             8cbbcd              TO BT BR BR BT SA  
       25             8cbccd              TO BT BT BR BT SA              8cbcd                 TO BT BR BT SA  
        9              8bbcd                 TO BT BR BR SA             8cbbcd              TO BT BR BR BT SA  
        4             8cbbcd              TO BT BR BR BT SA              8cbcd                 TO BT BR BT SA  
        3            8cbbccd           TO BT BT BR BR BT SA              8cbcd                 TO BT BR BT SA  
        3               8ccd                    TO BT BT SA            8cbbbcd           TO BT BR BR BR BT SA  
        2              8bbcd                 TO BT BR BR SA            8cbbbcd           TO BT BR BR BR BT SA  
        2              8cccd                 TO BT BT BT SA               8ccd                    TO BT BT SA  
        2               8cbd                    TO BR BT SA               8ccd                    TO BT BT SA  
        1            8cbbbcd           TO BT BR BR BR BT SA             8cbbcd              TO BT BR BR BT SA  
        1              8cbcd                 TO BT BR BT SA             8cbbcd              TO BT BR BR BT SA  
        1               8ccd                    TO BT BT SA             86cbcd              TO BT BR BT SC SA  
        1           8cbbbccd        TO BT BT BR BR BR BT SA              8cbcd                 TO BT BR BT SA  
    .


* too many to list on commandline mask argument, so save to file and load that 

::

     In [3]: np.save("/tmp/ts19tag5mask_maligned.npy", ab.maligned.astype(np.uint32))       ## NB load fails if do not use  np.uint32

::

    ts 19 --pfx scan-ts-utail --mask /tmp/ts19tag5mask_maligned.npy                         ## fails because mask indices go beyond the base
    ts 19 --pfx scan-ts-utail --mask /tmp/ts19tag5mask_maligned.npy --generateoverride -1   ## works : all 799 photons are maligned


Viz everything together, too confusing to glean much::

    tv 19 --pfx scan-ts-utail 
    tv 19 --pfx scan-ts-utail  --vizg4


Select just the 161::

    In [15]: np.where(np.logical_and( a.seqhis == 0x8bbcd, b.seqhis == 0x8cbcd ))
    Out[15]: 
    (array([  1016,   5985,  14233,  14633,  15370,  16913,  22269,  25702,  33369,  35537,  37501,  42845,  48468,  52700,  85871,  97214, 109455, 114617, 123244, 129246, 137050, 142250, 151713, 154477,
            158001, 158429, 159184, 163708, 171450, 186287, 200089, 208320, 208981, 211623, 228554, 239596, 252555, 255151, 268535, 274196, 279098, 279538, 291110, 291755, 297292, 317059, 317339, 322507,
            329729, 337281, 337572, 341416, 341978, 349357, 371055, 375034, 381344, 386810, 396248, 401396, 405544, 407334, 408919, 415443, 420581, 423282, 425678, 428930, 430608, 445853, 450327, 457313,
            461282, 463858, 466271, 470138, 476925, 479665, 488315, 493612, 497291, 505162, 507388, 510106, 511845, 512212, 515900, 524997, 525346, 525927, 527639, 530988, 538629, 539375, 542675, 549181,
            551198, 556371, 567609, 572023, 582188, 582528, 585733, 598252, 602251, 606590, 622589, 627033, 655016, 676326, 678625, 681954, 695260, 698423, 720481, 720652, 724573, 726543, 727246, 731863,
            743708, 747574, 761992, 763830, 770221, 779155, 782181, 793993, 796415, 806097, 816677, 828446, 832090, 834603, 841586, 846619, 857556, 859837, 859930, 870418, 876614, 876820, 882864, 883838,
            891715, 896452, 913217, 916891, 926609, 934457, 938475, 943150, 944819, 953069, 966489, 970035, 970631, 973732, 974034, 976993, 979276]),)

    In [16]: np.where(np.logical_and( a.seqhis == 0x8bbcd, b.seqhis == 0x8cbcd ))[0].shape
    Out[16]: (161,)

    In [17]: m161 = np.where(np.logical_and( a.seqhis == 0x8bbcd, b.seqhis == 0x8cbcd ))[0]

    In [18]: np.save("/tmp/ts19tag5mask_maligned_m161.npy", m161.astype(np.uint32))


::

    ts 19 --pfx scan-ts-utail --mask /tmp/ts19tag5mask_maligned_m161.npy --generateoverride -1   ## works : all 161 photons are maligned






migration A:"TO BT BR BR BT SA" B:"TO BT BR BR BR BT SA" at 7/1M level : suspect caused by different BR on ellipsoid/cone edge 
---------------------------------------------------------------------------------------------------------------------------------

* one more "BR" for G4
* vis checking it looks like the 2nd BR is on or very close to an edge   


Run on a single photon::

   DsG4OpBoundaryProcess=ERROR CSteppingAction=ERROR CRandomEngine=ERROR ts 19 --pfx scan-ts-utail --mask 2908 --pindex 0 --dbgseqhis 0xbbbbbbbbcd --pindexlog --recpoi --utaildebug --xanalytic --dbgflat

   tv 19 --pfx scan-ts-utail 
   tv 19 --pfx scan-ts-utail --vizg4  

   ta 19 --pfx scan-ts-utail 

::

    In [2]: a.rposta
    Out[2]: 
    A([[[  70.15  ,  -15.6827, -746.9043,    0.    ],          TO
        [  70.15  ,  -15.6827,  -17.1665,    2.4339],          BT    (on cone neck)
        [  -9.2453,    2.0773,  167.0085,    3.6543],          BR    (on flat top)  
        [ -81.9291,   18.3079,   -1.5066,    4.7706],          BR    <--- on ellipsoid/cone edge
        [ -67.4335,   15.0664,  167.0085,    5.7955],
        [  15.8654,   -3.5383,  746.9956,    7.7514],
        [   0.    ,    0.    ,    0.    ,    0.    ],
        [   0.    ,    0.    ,    0.    ,    0.    ],
        [   0.    ,    0.    ,    0.    ,    0.    ],
        [   0.    ,    0.    ,    0.    ,    0.    ]]])

    In [3]: b.rposta
    Out[3]: 
    A([[[  70.15  ,  -15.6827, -746.9043,    0.    ],
        [  70.15  ,  -15.6827,  -17.1665,    2.4339],
        [  -9.2453,    2.0773,  167.0085,    3.6543],
        [ -81.9291,   18.3079,   -1.5523,    4.771 ],                <--- on edge 
        [ 172.122 ,  -38.4877,   40.6564,    6.369 ],
        [ 236.5878,  -52.8921,  167.0085,    7.2328],
        [ 746.9956, -167.0085,  601.9474,    9.5019],
        [   0.    ,    0.    ,    0.    ,    0.    ],
        [   0.    ,    0.    ,    0.    ,    0.    ],
        [   0.    ,    0.    ,    0.    ,    0.    ]]])


x019 gives the param of ellipsoid and cone the intersection edge of which is being hit,
but thats difficult to calculate so instead select all maligned similarly:

::

    A:"TO BT BR [BR] BT SA" 
    B:"TO BT BR [BR] BR BT SA" 

and plot the 3D positions of the two 2nd BRs  


::

    TAG=3 ta 19 --pfx scan-ts-utail --msli :1M

    In [2]: a.rposta[2908]
    Out[2]: 
    A([[  70.15  ,  -15.6827, -746.9043,    0.    ],
       [  70.15  ,  -15.6827,  -17.1665,    2.4339],
       [  -9.2453,    2.0773,  167.0085,    3.6543],
       [ -81.9291,   18.3079,   -1.5066,    4.7706],   <<< 
       [ -67.4335,   15.0664,  167.0085,    5.7955],
       [  15.8654,   -3.5383,  746.9956,    7.7514],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ]])

    In [3]: b.rposta[2908]
    Out[3]: 
    A([[  70.15  ,  -15.6827, -746.9043,    0.    ],
       [  70.15  ,  -15.6827,  -17.1665,    2.4339],
       [  -9.2453,    2.0773,  167.0085,    3.6543],
       [ -81.9291,   18.3079,   -1.5523,    4.771 ],   <<<
       [ 172.122 ,  -38.4877,   40.6564,    6.369 ],
       [ 236.5878,  -52.8921,  167.0085,    7.2328],
       [ 746.9956, -167.0085,  601.9474,    9.5019],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ]])


::

    In [7]: a.seqhis[2908] == 0x8cbbcd
    Out[7]: True

    In [9]: b.seqhis[2908] == 0x8cbbbcd 
    Out[9]: True


    ## selecting photons with same history migration  : not enough to plot the edge 

    In [18]: np.where( np.logical_and( a.seqhis == 0x8cbbcd, b.seqhis == 0x8cbbbcd ) )
    Out[18]: (array([  2908, 270732, 361909, 572061, 709784, 813069, 880949]),)


    In [19]: a.rposta[270732]
    Out[19]: 
    A([[ -71.7707,    4.5884, -746.9043,    0.    ],
       [ -71.7707,    4.5884,  -17.1209,    2.4344],
       [   9.4051,   -0.5935,  167.0085,    3.6543],
       [  83.7325,   -5.3417,   -1.5295,    4.7706],    <<<
       [  68.9629,   -4.4058,  167.0085,    5.7955],
       [ -15.9338,    1.0273,  746.9956,    7.751 ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ]])

    In [20]: b.rposta[270732]
    Out[20]: 
    A([[ -71.7707,    4.5884, -746.9043,    0.    ],
       [ -71.7707,    4.5884,  -17.1209,    2.4344],
       [   9.4051,   -0.5935,  167.0085,    3.6543],
       [  83.7554,   -5.3417,   -1.5979,    4.771 ],    <<<
       [-175.9342,   11.2313,   40.5879,    6.3685],
       [-242.0208,   15.4316,  167.0085,    7.2332],
       [-746.9956,   47.6417,  586.0821,    9.4247],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ]])


    In [25]: mig = np.where( np.logical_and( a.seqhis == 0x8cbbcd, b.seqhis == 0x8cbbbcd ) )


    ## getting repeated shifts in the BR position, note all dz +ve 

    In [31]: for _ in mig[0]:print("%s \n %r " % (_,a.rposta[_]-b.rposta[_]))
    2908 
     A([[   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.0457,   -0.0005],
       [-239.5554,   53.5541,  126.3521,   -0.5734],
       [-220.7224,   49.3538,  579.9871,    0.5186],
       [-746.9956,  167.0085, -601.9474,   -9.5019],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ]]) 
    270732 
     A([[   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [  -0.0228,    0.    ,    0.0685,   -0.0005],
       [ 244.8971,  -15.6371,  126.4206,   -0.573 ],
       [ 226.087 ,  -14.4044,  579.9871,    0.5177],
       [ 746.9956,  -47.6417, -586.0821,   -9.4247],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ]]) 
    361909 
     A([[   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,   -0.0228,    0.0685,   -0.0005],
       [  92.2701,  227.3882,  126.4206,   -0.573 ],
       [  85.1935,  209.902 ,  579.9871,    0.5177],
       [ 303.131 ,  746.9956, -633.8608,   -9.6744],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ]]) 
    572061 
     A([[   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.0228,    0.    ],
       [ 131.6254, -207.254 ,  126.3293,   -0.5739],
       [ 121.1474, -190.7495,  579.9871,    0.5186],
       [ 474.3626, -746.9956, -702.4584,  -10.0228],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ]]) 
    709784 
     A([[   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [  -0.0228,   -0.0228,    0.0685,   -0.0005],
       [ 240.4685,   48.8972,  126.4206,   -0.573 ],
       [ 221.978 ,   45.1306,  579.9871,    0.5177],
       [ 746.9956,  151.9193, -597.5417,   -9.4845],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ]]) 
    813069 
     A([[   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.0228,    0.0685,   -0.0005],
       [-146.0754, -197.2098,  126.3978,   -0.573 ],
       [-134.7984, -182.0064,  579.9871,    0.5177],
       [-553.2328, -746.9956, -736.3121,  -10.2095],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ]]) 
    880949 
     A([[   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,   -0.0457,    0.0685,   -0.0005],
       [ -53.5998,  239.4641,  126.4206,   -0.573 ],
       [ -49.4908,  221.0649,  579.9871,    0.5177],
       [-167.214 ,  746.9956, -600.1212,   -9.4982],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ]]) 




G4 extra BT migration, A:"TO BT BT BT SA" B:"TO BT BT BT BT SA"  np.where( np.logical_and( a.seqhis == 0x8cccd, b.seqhis == 0x8ccccd ) )
-----------------------------------------------------------------------------------------------------------------------------------------------

* A:tis straight thru
* B:manages a duplicated point after which are thrown off in x,y  


::

          1   4860 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
          2   5477 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 


::

    In [33]: mig2 = np.where( np.logical_and( a.seqhis == 0x8cccd, b.seqhis == 0x8ccccd ) )

    In [26]: mig2[0].shape
    Out[26]: (105,)


    In [34]: mig2
    Out[34]: 
    (array([  4860,   5477,  18117,  32764,  37671,  43675,  45874,  46032,  60178,  63381,  72351,  78372,  86277, 114621, 114824, 117993, 124178, 128075, 130718, 190297, 203736, 218071, 228843, 230351,
            250888, 256617, 267200, 276009, 278342, 283215, 291654, 313341, 327592, 338295, 341215, 341963, 345982, 367144, 373297, 377105, 394971, 402159, 403227, 405408, 416093, 417813, 427720, 428899,
            431060, 476390, 507790, 546521, 558254, 573070, 578716, 582418, 597307, 598794, 603465, 628934, 639872, 648324, 666756, 670489, 676237, 693859, 702375, 708269, 713277, 721869, 734721, 741542,
            749771, 752564, 760508, 776795, 826254, 828763, 833845, 849092, 850833, 858629, 866552, 868538, 869816, 880413, 883392, 895631, 896158, 899711, 903051, 903983, 910087, 916302, 924279, 924566,
            924693, 935967, 936440, 943737, 945939, 948098, 953307, 991005, 995200]),)

    In [49]: "--mask="+",".join(map(str,mig2[0]))
    Out[49]: '--mask=4860,5477,18117,32764,37671,43675,45874,46032,60178,63381,72351,78372,86277,114621,114824,117993,124178,128075,130718,190297,203736,218071,228843,230351,250888,256617,267200,276009,278342,283215,291654,313341,327592,338295,341215,341963,345982,367144,373297,377105,394971,402159,403227,405408,416093,417813,427720,428899,431060,476390,507790,546521,558254,573070,578716,582418,597307,598794,603465,628934,639872,648324,666756,670489,676237,693859,702375,708269,713277,721869,734721,741542,749771,752564,760508,776795,826254,828763,833845,849092,850833,858629,866552,868538,869816,880413,883392,895631,896158,899711,903051,903983,910087,916302,924279,924566,924693,935967,936440,943737,945939,948098,953307,991005,995200'



    In [35]: mig3 = np.where( np.logical_and( a.seqhis == 0x8ccccd, b.seqhis == 0x8cccd ) )      ## one way migration 

    In [36]: mig3
    Out[36]: (array([], dtype=int64),)


    In [37]: a.rposta[4860]
    Out[37]: 
    A([[ -15.386 ,  -42.3   , -746.9043,    0.    ],
       [ -15.386 ,  -42.3   , -167.0085,    1.9344],
       [ -15.386 ,  -42.3   ,  -51.979 ,    2.6311],
       [ -15.386 ,  -42.3   ,  167.0085,    3.9583],
       [ -15.386 ,  -42.3   ,  746.9956,    5.8928],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ]])

    In [38]: b.rposta[4860]
    Out[38]: 
    A([[ -15.386 ,  -42.3   , -746.9043,    0.    ],
       [ -15.386 ,  -42.3   , -167.0085,    1.9344],
       [ -15.386 ,  -42.3   ,  -51.979 ,    2.6311],  <<<< how did a duplicate happen ?
       [ -15.386 ,  -42.3   ,  -51.979 ,    2.6311],
       [  17.6916,   48.6233,  167.0085,    4.0821],
       [ 196.9358,  541.6135,  746.9956,    6.6904],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ],
       [   0.    ,    0.    ,    0.    ,    0.    ]])







    In [44]: for _ in mig2[0]:print(_,"\n", a.rposta[_], "\n", b.rposta[_])
    4860 
     [[ -15.386   -42.3    -746.9043    0.    ]
     [ -15.386   -42.3    -167.0085    1.9344]
     [ -15.386   -42.3     -51.979     2.6311]
     [ -15.386   -42.3     167.0085    3.9583]
     [ -15.386   -42.3     746.9956    5.8928]
     [   0.        0.        0.        0.    ]
     [   0.        0.        0.        0.    ]
     [   0.        0.        0.        0.    ]
     [   0.        0.        0.        0.    ]
     [   0.        0.        0.        0.    ]] 
     [[ -15.386   -42.3    -746.9043    0.    ]
     [ -15.386   -42.3    -167.0085    1.9344]
     [ -15.386   -42.3     -51.979     2.6311]
     [ -15.386   -42.3     -51.979     2.6311]
     [  17.6916   48.6233  167.0085    4.0821]
     [ 196.9358  541.6135  746.9956    6.6904]
     [   0.        0.        0.        0.    ]
     [   0.        0.        0.        0.    ]
     [   0.        0.        0.        0.    ]
     [   0.        0.        0.        0.    ]]
    5477 
     [[  -9.154    44.0577 -746.9043    0.    ]
     [  -9.154    44.0577 -167.0085    1.9344]
     [  -9.154    44.0577  -51.979     2.6311]
     [  -9.154    44.0577  167.0085    3.9583]
     [  -9.154    44.0577  746.9956    5.8928]
     [   0.        0.        0.        0.    ]
     [   0.        0.        0.        0.    ]
     [   0.        0.        0.        0.    ]
     [   0.        0.        0.        0.    ]
     [   0.        0.        0.        0.    ]] 
     [[  -9.154    44.0577 -746.9043    0.    ]
     [  -9.154    44.0577 -167.0085    1.9344]
     [  -9.154    44.0577  -51.979     2.6311]
     [  -9.154    44.0577  -51.979     2.6311]
     [  10.5236  -50.655   167.0085    4.0821]
     [ 117.2666 -564.2587  746.9956    6.6904]
     [   0.        0.        0.        0.    ]
     [   0.        0.        0.        0.    ]
     [   0.        0.        0.        0.    ]
     [   0.        0.        0.        0.    ]]
    18117 
     [[  30.5208   33.0775 -746.9043    0.    ]
     [  30.5208   33.0775 -167.0085    1.9344]
     [  30.5208   33.0775  -51.979     2.6311]
     [  30.5208   33.0775  167.0085    3.9583]
     [  30.5208   33.0775  746.9956    5.8928]
     [   0.        0.        0.        0.    ]
     [   0.        0.        0.        0.    ]
     [   0.        0.        0.        0.    ]
     [   0.        0.        0.        0.    ]
     [   0.        0.        0.        0.    ]] 
     [[  30.5208   33.0775 -746.9043    0.    ]
     [  30.5208   33.0775 -167.0085    1.9344]
     [  30.5208   33.0775  -51.979     2.6311]
     [  30.5208   33.0775  -51.979     2.6311]
     [ -35.1092  -38.0084  167.0085    4.0821]
     [-390.9497 -423.4337  746.9956    6.6904]
     [   0.        0.        0.        0.    ]
     [   0.        0.        0.        0.    ]
     [   0.        0.        0.        0.    ]
     [   0.        0.        0.        0.    ]]



run on just these migrated (order 40/1M)
----------------------------------------------

* they are all in a ring that matches the tube  


::

   ts 19 --pfx scan-ts-utail --mask=4860,5477,18117,32764,37671,43675,45874,46032,60178,63381,72351,78372,86277,114621,114824,117993,124178,128075,130718,190297,203736,218071,228843,230351,250888,256617,267200,276009,278342,283215,291654,313341,327592,338295,341215,341963,345982,367144,373297,377105,394971,402159,403227,405408,416093,417813,427720,428899,431060,476390,507790,546521,558254,573070,578716,582418,597307,598794,603465,628934,639872,648324,666756,670489,676237,693859,702375,708269,713277,721869,734721,741542,749771,752564,760508,776795,826254,828763,833845,849092,850833,858629,866552,868538,869816,880413,883392,895631,896158,899711,903051,903983,910087,916302,924279,924566,924693,935967,936440,943737,945939,948098,953307,991005,995200  --generateoverride -1 

   tv 19 --pfx scan-ts-utail

       ## straight thru 

   tv 19 --pfx scan-ts-utail --vizg4

        ## very different, the photons around the ring skirt the outside of the base tube and get diverted inwards 
        ## on hitting the neck  



check 10k unbiased : get 3 maligned which look to all be edge effects
-------------------------------------------------------------------------

::

    TAG=4 ts 19 --pfx scan-ts-utail 
 
    .
    ab.mal
    aligned     9997/  10000 : 0.9997 : 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 
    maligned       3/  10000 : 0.0003 : 2908,4860,5477 
    slice(0, 25, None)
          0   2908 : * :                                  TO BT BR BR BT SA                               TO BT BR BR BR BT SA 
          1   4860 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
          2   5477 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
    .


