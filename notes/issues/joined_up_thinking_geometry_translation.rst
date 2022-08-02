joined_up_thinking_geometry_translation
==========================================

Current geometry translation has historical baggage, and lots more code than necessary
as it has a complete geometry model in the middle::

    Geant4 -> GGeo/NPY  -> CSGFoundry 


How to handle double precision instance transforms in CSGFoundry ?
----------------------------------------------------------------------

* collect and persist in double precision
* narrow to float only transiently in memory just before uploading 
* so the float precision transforms only actually get used on device  
* dqat4 sibling of qat4 for use on host 


Explorations 
---------------

u4/U4Transform.h
    get object and frame transforms from PV 
    
u4/tests/U4TransformTest.cc
    test getting transforms from PV and writing them into NP arrays 
    or glm::tmat4x4<double> 

u4/U4Tree.h 
    thinking about minimal structure translation

    * is serialization of the intermediate tree needed ? 
    * n-ary tree serialization turns out to be straightforward using 
      nidx links : firstchild/next_sibling/parent 

sysrap/stree.h 
sysrap/sfreq.h 
sysrap/snode.h
   * persisting n-ary tree including double precision transforms with minimal dependency  
   

Tree background
--------------------

* https://hbfs.wordpress.com/2009/04/07/compact-tree-storage/

* https://hbfs.wordpress.com/2016/09/06/serializing-trees/#1



TODO : compare stree_test with GGeo 
---------------------------------------

* need to arrange to get the same set of repeats in same order and verify 
  that the transforms match 

::

    [ stree::disqualifyContainedRepeats 
    ] stree::disqualifyContainedRepeats  disqualify.size 23
    [ stree::sortSubtrees 
    ] stree::sortSubtrees 
    st.desc_sub
        0 : c2520d0897b02efe301aed3f8d8b41e8 : 32256 de:( 9  9) 1st:    17 sBar0x71a9200
        1 : 246cf1cae2a2304dad8dbafa5238934f : 25600 de:( 6  6) 1st:194249 PMT_3inch_pmt_solid0x66e59f0
        2 : e238e3e830cc4e95eb9b167c54d155a2 : 12612 de:( 6  6) 1st: 70965 NNVTMCPPMTsMask_virtual0x5f5f900
        3 : 881ef0f2f7f79f81479dd6e0a07a380b :  5000 de:( 6  6) 1st: 70972 HamamatsuR12860sMask_virtual0x5f50d40
        4 : 25ed11817b62fa562aaef3daba337336 :  2400 de:( 4  4) 1st:322253 mask_PMT_20inch_vetosMask_virtual0x5f62e40
        5 : c051c1bb98b71ccb15b0cf9c67d143ee :   590 de:( 6  6) 1st: 68493 sStrutBallhead0x5853640
        6 : 5e01938acb3e0df0543697fc023bffb1 :   590 de:( 6  6) 1st: 69083 uni10x5832ff0
        7 : cdc824bf721df654130ed7447fb878ac :   590 de:( 6  6) 1st: 69673 base_steel0x58d3270
        8 : 3fd85f9ee7ca8882c8caa747d0eef0b3 :   590 de:( 6  6) 1st: 70263 uni_acrylic10x597c090
        9 : d4f5974d740cd7c78613c9d8563878c7 :   504 de:( 7  7) 1st:    15 sPanel0x71a8d90



* sBar is different ? Looks like instance inside instance inside instance ...
* this is why need to check more than just the parent for contained repeat 

::

    snode ix:  65720 dh: 9 sx:   63 pt:  65593 nc:    1 fc:  65721 ns:     -1 lv:  9 sBar0x71a9200
    stree::desc_ancestry nidx 17
    snode ix:      0 dh: 0 sx:   -1 pt:     -1 nc:    2 fc:      1 ns:     -1 lv:138    92 : 429a9f424f2e67d955836ecc49249c06 :     1 sWorld0x577e4d0
    snode ix:      1 dh: 1 sx:    0 pt:      0 nc:    2 fc:      2 ns:  65722 lv: 17    93 : 3f5a0d33e1ba4bfd47ecd77f7486f24f :     1 sTopRock0x578c0a0
    snode ix:      5 dh: 2 sx:    1 pt:      1 nc:    1 fc:      6 ns:     -1 lv: 16    97 : 01bdaba672bbda09bbafcb22487052ef :     1 sExpRockBox0x578ce00
    snode ix:      6 dh: 3 sx:    0 pt:      5 nc:    3 fc:      7 ns:     -1 lv: 15    98 : 7f8bfc13b2d2185223e50362e3416ba6 :     1 sExpHall0x578d4f0
    snode ix:     12 dh: 4 sx:    2 pt:      6 nc:   63 fc:     13 ns:     -1 lv: 14   104 : 9de4752996fe00065bbe29aa024161d1 :     1 sAirTT0x71a76a0
    snode ix:     13 dh: 5 sx:    0 pt:     12 nc:    2 fc:     14 ns:   1056 lv: 13    13 : 3d2cdc54d35c77630c06a2614d700410 :    63 sWall0x71a8b30
    snode ix:     14 dh: 6 sx:    0 pt:     13 nc:    4 fc:     15 ns:    535 lv: 12    12 : b6315f2ea7550a1ca922a1fc1c5102c3 :   126 sPlane0x71a8bb0
    snode ix:     15 dh: 7 sx:    0 pt:     14 nc:    1 fc:     16 ns:    145 lv: 11     9 : d4f5974d740cd7c78613c9d8563878c7 :   504 sPanel0x71a8d90
    snode ix:     16 dh: 8 sx:    0 pt:     15 nc:   64 fc:     17 ns:     -1 lv: 10   116 : 850bf8dcd5f6b272c13a49ac3f22f87d :  -504 sPanelTape0x71a9090

    snode ix:     17 dh: 9 sx:    0 pt:     16 nc:    1 fc:     18 ns:     19 lv:  9     0 : c2520d0897b02efe301aed3f8d8b41e8 : 32256 sBar0x71a9200 


* note that sBar is inside sPanel 


HMM : the totals "63 sWall0x71a8b30" are for entire geometry...

* need to examine those within single subtrees and see the extents and transforms to work out whats
  appropriate 
  

How to handle repeats inside repeats ? Which level to treat as the factor ?
-------------------------------------------------------------------------------

* for now : just need to duplicate what GGeo did
* but potentially this is an area to optimize based on the extent of the different choices

  * compound solid instances must not be too large : as that prevents acceleration structure from working 
  * also presumbly should not be too small either : due to instancing overheads ?
  * where to draw the line requires performance measurement


Dump ancestry of first sBar::

    SO sBar lvs [8 9]
    lv:8 bb=st.find_lvid_nodes(lv)  bb:[   18    20    22    24    26 ... 65713 65715 65717 65719 65721] b:18 
    b:18 anc=st.get_ancestors(b) anc:[0, 1, 5, 6, 12, 13, 14, 15, 16, 17] 
    st.desc_nodes(anc, brief=True))
    +               snode ix:      0 dh: 0 nc:    2 lv:138. sf 138 :       1 : 8ab4541c79ebf0604ebe21f17db28154. sWorld0x577e4d0
     +              snode ix:      1 dh: 1 nc:    2 lv: 17. sf 118 :       1 : 6eaf94bd8099cb88c2a12ead92d51023. sTopRock0x578c0a0
      +             snode ix:      5 dh: 2 nc:    1 lv: 16. sf 120 :       1 : 6db9a04bca07b11d86e13b4fba6443f7. sExpRockBox0x578ce00
       +            snode ix:      6 dh: 3 nc:    3 lv: 15. sf 121 :       1 : 3736e587ccbbcdc3244e7d793841f355. sExpHall0x578d4f0
        +           snode ix:     12 dh: 4 nc:   63 lv: 14. sf 127 :       1 : f6323ce1038bd8d00e1924e6398f3932. sAirTT0x71a76a0
         +          snode ix:     13 dh: 5 nc:    2 lv: 13. sf  36 :      63 : 66a4f10f1cd4c573daa1fcf639046c98. sWall0x71a8b30
          +         snode ix:     14 dh: 6 nc:    4 lv: 12. sf  35 :     126 : 09a56a00c622c585bb012e8f270cca70. sPlane0x71a8bb0
           +        snode ix:     15 dh: 7 nc:    1 lv: 11. sf  32 :     504 : 7d9a644fae10bdc1899c0765077e7a33. sPanel0x71a8d90
            +       snode ix:     16 dh: 8 nc:   64 lv: 10. sf  31 :     504 : a1a353cb718ab2a987b199f46da1699f. sPanelTape0x71a9090
             +      snode ix:     17 dh: 9 nc:    1 lv:  9. sf   1 :   32256 : 72cfbefa2b68ea3cbe1e682aba2ae38e. sBar0x71a9200
    st.desc_nodes([b], brief=True))
              +     snode ix:     18 dh:10 nc:    0 lv:  8. sf   0 :   32256 : 34f45818f16d1bbb62ba5874b8814cc7. sBar0x71a9370


* CAUTION: with the meaning of the sf digest counts : they are for entire geometry : not single subtree which is more intuitive 


Current GGeo factorizes at sPanel : due to repeat cut of 500
-----------------------------------------------------------------

* reason that factorization happens at sPanel is  apparent from the counts and the repeat cut of 500
* "sf:63 sWall" "sf:126 sPlane" have too few subtree digests to become factors   


::

    epsilon:CSGFoundry blyth$ cat mmlabel.txt 
    3089:sWorld
    5:PMT_3inch_pmt_solid
    7:NNVTMCPPMTsMask_virtual
    7:HamamatsuR12860sMask_virtual
    6:mask_PMT_20inch_vetosMask_virtual
    1:sStrutBallhead
    1:uni1
    1:base_steel
    1:uni_acrylic1
    130:sPanel
    epsilon:CSGFoundry blyth$ 


DONE: the "130:sPanel" : the 130 is 1+the number of progeny of each sPanel::

    In [11]: pp = st.find_lvid_nodes("sPanel0x")
    In [12]: len(pp) 
    Out[12]: 504

    In [2]: pp_0 = st.get_progeny(pp[0])

    In [4]: len(pp_0)    ## 129 = 64*2 + 1       
    Out[4]: 129

    In [3]: print(st.desc_nodes(pp_0, brief=True))
            +       snode ix:     16 dh: 8 nc:   64 lv: 10. sf  31 :     504 : a1a353cb718ab2a987b199f46da1699f. sPanelTape0x71a9090
             +      snode ix:     17 dh: 9 nc:    1 lv:  9. sf   1 :   32256 : 72cfbefa2b68ea3cbe1e682aba2ae38e. sBar0x71a9200
             +      snode ix:     19 dh: 9 nc:    1 lv:  9. sf   1 :   32256 : 72cfbefa2b68ea3cbe1e682aba2ae38e. sBar0x71a9200
             +      snode ix:     21 dh: 9 nc:    1 lv:  9. sf   1 :   32256 : 72cfbefa2b68ea3cbe1e682aba2ae38e. sBar0x71a9200
             +      snode ix:     23 dh: 9 nc:    1 lv:  9. sf   1 :   32256 : 72cfbefa2b68ea3cbe1e682aba2ae38e. sBar0x71a9200
             +      snode ix:     25 dh: 9 nc:    1 lv:  9. sf   1 :   32256 : 72cfbefa2b68ea3cbe1e682aba2ae38e. sBar0x71a9200
             +      snode ix:     27 dh: 9 nc:    1 lv:  9. sf   1 :   32256 : 72cfbefa2b68ea3cbe1e682aba2ae38e. sBar0x71a9200
             +      snode ix:     29 dh: 9 nc:    1 lv:  9. sf   1 :   32256 : 72cfbefa2b68ea3cbe1e682aba2ae38e. sBar0x71a9200
             +      snode ix:     31 dh: 9 nc:    1 lv:  9. sf   1 :   32256 : 72cfbefa2b68ea3cbe1e682aba2ae38e. sBar0x71a9200
             +      snode ix:     33 dh: 9 nc:    1 lv:  9. sf   1 :   32256 : 72cfbefa2b68ea3cbe1e682aba2ae38e. sBar0x71a9200
             +      snode ix:     35 dh: 9 nc:    1 lv:  9. sf   1 :   32256 : 72cfbefa2b68ea3cbe1e682aba2ae38e. sBar0x71a9200
             +      snode ix:     37 dh: 9 nc:    1 lv:  9. sf   1 :   32256 : 72cfbefa2b68ea3cbe1e682aba2ae38e. sBar0x71a9200
             +      snode ix:     39 dh: 9 nc:    1 lv:  9. sf   1 :   32256 : 72cfbefa2b68ea3cbe1e682aba2ae38e. sBar0x71a9200
             +      snode ix:     41 dh: 9 nc:    1 lv:  9. sf   1 :   32256 : 72cfbefa2b68ea3cbe1e682aba2ae38e. sBar0x71a9200



TODO : look at how CSG_GGeo serializes the transforms into CSG and mimic that for comparison
----------------------------------------------------------------------------------------------






HMM: maybe factorizing at sWall would be cleanest : because its the higest repeater : but only 63
------------------------------------------------------------------------------------------------------

* the criteria should probably also consider how much progeny is inside the subtree : not 
  just how many times the subtree is repeated

The advantage of factorizing higher up the tree is that it mops up loadsa nodes that no longer get repeated in the global remainder. 


::

    In [1]: ww = st.find_lvid_nodes("sWall")                                                                                                                                                         

    In [2]: ww                                                                                                                                                                                       
    Out[2]: 
    array([   13,  1056,  2099,  3142,  4185,  5228,  6271,  7314,  8357,  9400, 10443, 11486, 12529, 13572, 14615, 15658, 16701, 17744, 18787, 19830, 20873, 21916, 22959, 24002, 25045, 26088, 27131,
           28174, 29217, 30260, 31303, 32346, 33389, 34432, 35475, 36518, 37561, 38604, 39647, 40690, 41733, 42776, 43819, 44862, 45905, 46948, 47991, 49034, 50077, 51120, 52163, 53206, 54249, 55292,
           56335, 57378, 58421, 59464, 60507, 61550, 62593, 63636, 64679])

    In [3]: len(ww)                                                                                                                                                                                  
    Out[3]: 63


    In [13]: print(st.desc_nodes(ww))                                                                                                                                                                 
         +          snode ix:     13 dh: 5 sx:    0 pt:     12 nc:    2 fc:     14 ns:   1056 lv: 13 cp:      0. sf  36 :      63 : 66a4f10f1cd4c573daa1fcf639046c98. sWall0x71a8b30
         +          snode ix:   1056 dh: 5 sx:    1 pt:     12 nc:    2 fc:   1057 ns:   2099 lv: 13 cp:      0. sf  36 :      63 : 66a4f10f1cd4c573daa1fcf639046c98. sWall0x71a8b30
         +          snode ix:   2099 dh: 5 sx:    2 pt:     12 nc:    2 fc:   2100 ns:   3142 lv: 13 cp:      0. sf  36 :      63 : 66a4f10f1cd4c573daa1fcf639046c98. sWall0x71a8b30
         +          snode ix:   3142 dh: 5 sx:    3 pt:     12 nc:    2 fc:   3143 ns:   4185 lv: 13 cp:      0. sf  36 :      63 : 66a4f10f1cd4c573daa1fcf639046c98. sWall0x71a8b30
         +          snode ix:   4185 dh: 5 sx:    4 pt:     12 nc:    2 fc:   4186 ns:   5228 lv: 13 cp:      0. sf  36 :      63 : 66a4f10f1cd4c573daa1fcf639046c98. sWall0x71a8b30
         +          snode ix:   5228 dh: 5 sx:    5 pt:     12 nc:    2 fc:   5229 ns:   6271 lv: 13 cp:      0. sf  36 :      63 : 66a4f10f1cd4c573daa1fcf639046c98. sWall0x71a8b30
         +          snode ix:   6271 dh: 5 sx:    6 pt:     12 nc:    2 fc:   6272 ns:   7314 lv: 13 cp:      0. sf  36 :      63 : 66a4f10f1cd4c573daa1fcf639046c98. sWall0x71a8b30


    In [9]: st.get_children(ww[0])                                                                                                                                                                    
    Out[9]: [14, 535]

    In [10]: ww_0 = st.get_children(ww[0])                                                                                                                                                            

    In [11]: print(st.desc_nodes(ww_0))                                                                                                                                                               
          +         snode ix:     14 dh: 6 sx:    0 pt:     13 nc:    4 fc:     15 ns:    535 lv: 12 cp:      0. sf  35 :     126 : 09a56a00c622c585bb012e8f270cca70. sPlane0x71a8bb0
          +         snode ix:    535 dh: 6 sx:    1 pt:     13 nc:    4 fc:    536 ns:     -1 lv: 12 cp:      1. sf  35 :     126 : 09a56a00c622c585bb012e8f270cca70. sPlane0x71a8bb0





    In [4]: st.f.trs[ww].reshape(-1,16)                                                                                                                                                              
    Out[4]: 
    array([[     0. ,      1. ,      0. ,      0. ,     -1. , ...,      0. ,  20133.6,  -6711.2,  -2455. ,      1. ],
           [     0. ,      1. ,      0. ,      0. ,     -1. , ...,      0. ,  13422.4,  -6711.2,  -2505. ,      1. ],
           [     0. ,      1. ,      0. ,      0. ,     -1. , ...,      0. ,   6711.2,  -6711.2,  -2455. ,      1. ],
           [     0. ,      1. ,      0. ,      0. ,     -1. , ...,      0. ,      0. ,  -6711.2,  -2505. ,      1. ],
           [     0. ,      1. ,      0. ,      0. ,     -1. , ...,      0. ,  -6711.2,  -6711.2,  -2455. ,      1. ],
           ...,
           [     0. ,      1. ,      0. ,      0. ,     -1. , ...,      0. ,   6711.2,   6711.2,    545. ,      1. ],
           [     0. ,      1. ,      0. ,      0. ,     -1. , ...,      0. ,      0. ,   6711.2,    495. ,      1. ],
           [     0. ,      1. ,      0. ,      0. ,     -1. , ...,      0. ,  -6711.2,   6711.2,    545. ,      1. ],
           [     0. ,      1. ,      0. ,      0. ,     -1. , ...,      0. , -13422.4,   6711.2,    495. ,      1. ],
           [     0. ,      1. ,      0. ,      0. ,     -1. , ...,      0. , -20133.6,   6711.2,    545. ,      1. ]])

    In [5]: np.set_printoptions(edgeitems=100)                                                                                                                                                       


All the sWall rotations are the same, just an XY axis flip::

   In [7]: st.f.trs[ww[0]]                                                                                                                                                                          
    Out[7]: 
    array([[    0. ,     1. ,     0. ,     0. ],
           [   -1. ,     0. ,     0. ,     0. ],
           [    0. ,     0. ,     1. ,     0. ],
           [20133.6, -6711.2, -2455. ,     1. ]])

    In [8]: st.f.trs[ww,3]                                                                                                                                                                           
    Out[8]: 
    array([[ 20133.6,  -6711.2,  -2455. ,      1. ],
           [ 13422.4,  -6711.2,  -2505. ,      1. ],
           [  6711.2,  -6711.2,  -2455. ,      1. ],
           [     0. ,  -6711.2,  -2505. ,      1. ],
           [ -6711.2,  -6711.2,  -2455. ,      1. ],
           [-13422.4,  -6711.2,  -2505. ,      1. ],
           [-20133.6,  -6711.2,  -2455. ,      1. ],
           [ 20133.6,      0. ,  -2555. ,      1. ],
           [ 13422.4,      0. ,  -2405. ,      1. ],
           [  6711.2,      0. ,  -2555. ,      1. ],
           [     0. ,      0. ,   3660. ,      1. ],
           [ -6711.2,      0. ,  -2555. ,      1. ],
           [-13422.4,      0. ,  -2405. ,      1. ],
           [-20133.6,      0. ,  -2555. ,      1. ],
           [ 20133.6,   6711.2,  -2455. ,      1. ],
           [ 13422.4,   6711.2,  -2505. ,      1. ],
           [  6711.2,   6711.2,  -2455. ,      1. ],
           [     0. ,   6711.2,  -2505. ,      1. ],
           [ -6711.2,   6711.2,  -2455. ,      1. ],
           [-13422.4,   6711.2,  -2505. ,      1. ],
           [-20133.6,   6711.2,  -2455. ,      1. ],
           [ 20133.6,  -6711.2,   -955. ,      1. ],
           [ 13422.4,  -6711.2,  -1005. ,      1. ],
           [  6711.2,  -6711.2,   -955. ,      1. ],
           [     0. ,  -6711.2,  -1005. ,      1. ],
           [ -6711.2,  -6711.2,   -955. ,      1. ],
           [-13422.4,  -6711.2,  -1005. ,      1. ],
           [-20133.6,  -6711.2,   -955. ,      1. ],
           [ 20133.6,      0. ,  -1055. ,      1. ],
           [ 13422.4,      0. ,   -905. ,      1. ],
           [  6711.2,      0. ,  -1055. ,      1. ],
           [     0. ,      0. ,   3910. ,      1. ],
           [ -6711.2,      0. ,  -1055. ,      1. ],
           [-13422.4,      0. ,   -905. ,      1. ],
           [-20133.6,      0. ,  -1055. ,      1. ],
           [ 20133.6,   6711.2,   -955. ,      1. ],
           [ 13422.4,   6711.2,  -1005. ,      1. ],
           [  6711.2,   6711.2,   -955. ,      1. ],
           [     0. ,   6711.2,  -1005. ,      1. ],
           [ -6711.2,   6711.2,   -955. ,      1. ],
           [-13422.4,   6711.2,  -1005. ,      1. ],
           [-20133.6,   6711.2,   -955. ,      1. ],
           [ 20133.6,  -6711.2,    545. ,      1. ],
           [ 13422.4,  -6711.2,    495. ,      1. ],
           [  6711.2,  -6711.2,    545. ,      1. ],
           [     0. ,  -6711.2,    495. ,      1. ],
           [ -6711.2,  -6711.2,    545. ,      1. ],
           [-13422.4,  -6711.2,    495. ,      1. ],
           [-20133.6,  -6711.2,    545. ,      1. ],
           [ 20133.6,      0. ,    445. ,      1. ],
           [ 13422.4,      0. ,    595. ,      1. ],
           [  6711.2,      0. ,    445. ,      1. ],
           [     0. ,      0. ,   4160. ,      1. ],
           [ -6711.2,      0. ,    445. ,      1. ],
           [-13422.4,      0. ,    595. ,      1. ],
           [-20133.6,      0. ,    445. ,      1. ],
           [ 20133.6,   6711.2,    545. ,      1. ],
           [ 13422.4,   6711.2,    495. ,      1. ],
           [  6711.2,   6711.2,    545. ,      1. ],
           [     0. ,   6711.2,    495. ,      1. ],
           [ -6711.2,   6711.2,    545. ,      1. ],
           [-13422.4,   6711.2,    495. ,      1. ],
           [-20133.6,   6711.2,    545. ,      1. ]])


TripletIdentity
-----------------

::

    epsilon:CSG blyth$ opticks-f TripletIdentity 
    ./ggeo/GGeoTest.cc:        volume->setTripletIdentity(tripletIdentity); 
    ./ggeo/GNode.cc:GNode::setTripletIdentity
    ./ggeo/GNode.cc:void GNode::setTripletIdentity(unsigned triplet_identity)
    ./ggeo/GNode.cc:unsigned GNode::getTripletIdentity() const 
    ./ggeo/GInstancer.cc:    node->setTripletIdentity( triplet_identity ); 
    ./ggeo/GInstancer.cc:    node->setTripletIdentity( triplet_identity ); 
    ./ggeo/GVolume.cc:    glm::uvec4 id(getIndex(), getTripletIdentity(), getShapeIdentity(), getSensorIndex()) ; 
    ./ggeo/GMergedMesh.cc:    unsigned tripletIdentity = volume->getTripletIdentity(); 
    ./ggeo/GNode.hh:     void     setTripletIdentity(unsigned triplet_identity);
    ./ggeo/GNode.hh:     unsigned getTripletIdentity() const ;  
    ./optixrap/cu/generate.cu:    258     glm::uvec4 id(getIndex(), getTripletIdentity(), getShapeIdentity(), getSensorIndex()) ;
    epsilon:opticks blyth$ 





nidx into instance transforms ?
--------------------------------

HMM: this is leading towards cutting out GGeo from the translation

* seems no point in shoe-horning this into GGeo + CSG_GGeo trans
* can do it straightforwardly with CSG_stree direct translation from stree.h model into CSGFoundry model 
* HMM: should stree folder be persisted as sibling of CSGFoundry folder or within it ? 
* note that CSG already depends on sysrap : so CSG package can itself do the translation 


::

    1547 void CSGFoundry::addInstance(const float* tr16, unsigned gas_idx, unsigned ias_idx )
    1548 {
    1549     qat4 instance(tr16) ;  // identity matrix if tr16 is nullptr 
    1550     unsigned ins_idx = inst.size() ;
    1551 
    1552     instance.setIdentity( ins_idx, gas_idx, ias_idx );


    0200 void CSG_GGeo_Convert::addInstances(unsigned repeatIdx )
     201 {
     202     unsigned nmm = ggeo->getNumMergedMesh();
     203     assert( repeatIdx < nmm );
     204     const GMergedMesh* mm = ggeo->getMergedMesh(repeatIdx);
     205     unsigned num_inst = mm->getNumITransforms() ;
     206     NPY<unsigned>* iid = mm->getInstancedIdentityBuffer();
     207 
     208     LOG(LEVEL)
     209         << " repeatIdx " << repeatIdx
     210         << " num_inst (GMergedMesh::getNumITransforms) " << num_inst
     211         << " iid " << ( iid ? iid->getShapeString() : "-"  )
     212         ;
     213 
     214     //LOG(LEVEL) << " nmm " << nmm << " repeatIdx " << repeatIdx << " num_inst " << num_inst ; 
     215 
     216     for(unsigned i=0 ; i < num_inst ; i++)
     217     {
     218         glm::mat4 it = mm->getITransform_(i);
     219    
     220         const float* tr16 = glm::value_ptr(it) ;
     221         unsigned gas_idx = repeatIdx ;
     222         unsigned ias_idx = 0 ;
     223 
     224         foundry->addInstance(tr16, gas_idx, ias_idx);
     225     }
     226 }






DONE : Serialize n-ary tree
-----------------------------

* HMM by CSG list-nodes are related to this, should review them 

* https://www.geeksforgeeks.org/serialize-deserialize-n-ary-tree/

* :google:`tree serialization generic tree`

* https://eli.thegreenplace.net/2011/09/29/an-interesting-tree-serialization-algorithm-from-dwarf


Here's a quote from the DWARF v3 standard section 2.3 explaining it, slightly rephrased:

The tree itself is represented by flattening it in prefix order. Each node is
defined either to have children or not to have children. If a node is defined
not to have children, the next physically succeeding node is a sibling. If a
node is defined to have children, the next physically succeeding node is its
first child. Additional children are represented as siblings of the first
child. A chain of sibling entries is terminated by a null node.

 

