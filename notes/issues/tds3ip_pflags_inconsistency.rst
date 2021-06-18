tds3ip_pflags_inconsistency : FIXED BY SCRUBBING INITIAL PFLAGS FOR INPUT PHOTONS
===========================================================================================

clearing pflags of input photons avoids the obvious inconsistency
-------------------------------------------------------------------

::

    #ab.ahis
    ab.ahis
    .    all_seqhis_ana  cfo:sum  1:g4live:tds3ip   -1:g4live:tds3ip        c2        ab        ba 
    .                             100000    100000       325.20/13 = 25.02  (pval:0.000 prob:1.000)  
       n             iseq         a         b    a-b               c2          a/b                   b/a           [ns] label
    0000               8d     92759     92890   -131             0.09        0.999 +- 0.003        1.001 +- 0.003  [2 ] TO SA
    0001               4d      5997      5840    157             2.08        1.027 +- 0.013        0.974 +- 0.013  [2 ] TO AB
    0002             7c6d       253       302    -49             4.33        0.838 +- 0.053        1.194 +- 0.069  [4 ] TO SC BT SD
    0003              86d       200       240    -40             3.64        0.833 +- 0.059        1.200 +- 0.077  [3 ] TO SC SA
    0004            4cc6d        56       188   -132            71.41        0.298 +- 0.040        3.357 +- 0.245  [5 ] TO SC BT BT AB
    0005              46d       144        56     88            38.72        2.571 +- 0.214        0.389 +- 0.052  [3 ] TO SC AB
    0006             8c6d        80        83     -3             0.06        0.964 +- 0.108        1.038 +- 0.114  [4 ] TO SC BT SA
    0007           8cac6d        81         0     81            81.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO SC BT SR BT SA
    0008          8ccac6d         0        75    -75            75.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO SC BT SR BT BT SA
    0009             4c6d        40        19     21             7.47        2.105 +- 0.333        0.475 +- 0.109  [4 ] TO SC BT AB
    0010           46cc6d        16        38    -22             8.96        0.421 +- 0.105        2.375 +- 0.385  [6 ] TO SC BT BT SC AB
    0011        7ccc6cc6d        28         9     19             9.76        3.111 +- 0.588        0.321 +- 0.107  [9 ] TO SC BT BT SC BT BT BT SD
    0012            7cc6d         5        31    -26            18.78        0.161 +- 0.072        6.200 +- 1.114  [5 ] TO SC BT BT SD
    0013         7ccccc6d        21        10     11             3.90        2.100 +- 0.458        0.476 +- 0.151  [8 ] TO SC BT BT BT BT BT SD
    0014          466cc6d        11        17     -6             0.00        0.647 +- 0.195        1.545 +- 0.375  [7 ] TO SC BT BT SC SC AB
    0015            8cc6d        10        14     -4             0.00        0.714 +- 0.226        1.400 +- 0.374  [5 ] TO SC BT BT SA
    0016           4ccc6d         7        17    -10             0.00        0.412 +- 0.156        2.429 +- 0.589  [6 ] TO SC BT BT BT AB
    0017       ccacccac6d         0        24    -24             0.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO SC BT SR BT BT BT SR BT BT
    0018            7cb6d        10         9      1             0.00        1.111 +- 0.351        0.900 +- 0.300  [5 ] TO SC BR BT SD
    .                             100000    100000       325.20/13 = 25.02  (pval:0.000 prob:1.000)  
    #ab.flg
    ab.flg
    .       pflags_ana  cfo:sum  1:g4live:tds3ip   -1:g4live:tds3ip        c2        ab        ba 
    .                             100000    100000       139.94/10 = 13.99  (pval:0.000 prob:1.000)  
       n             iseq         a         b    a-b               c2          a/b                   b/a           [ns] label
    0000             1080     92759     92890   -131             0.09        0.999 +- 0.003        1.001 +- 0.003  [2 ] TO|SA
    0001             1008      5997      5840    157             2.08        1.027 +- 0.013        0.974 +- 0.013  [2 ] TO|AB
    0002             1828       169       310   -141            41.51        0.545 +- 0.042        1.834 +- 0.104  [4 ] TO|BT|SC|AB
    0003             10a0       201       245    -44             4.34        0.820 +- 0.058        1.219 +- 0.078  [3 ] TO|SA|SC
    0004             5860       181       215    -34             2.92        0.842 +- 0.063        1.188 +- 0.081  [5 ] EX|TO|BT|SD|SC
    0005             9860       136       159    -23             1.79        0.855 +- 0.073        1.169 +- 0.093  [5 ] EC|TO|BT|SD|SC
    0006             18a0       111       112     -1             0.00        0.991 +- 0.094        1.009 +- 0.095  [4 ] TO|BT|SA|SC
    0007             1028       145        57     88            38.34        2.544 +- 0.211        0.393 +- 0.052  [3 ] TO|SC|AB
    0008             1aa0        98        84     14             1.08        1.167 +- 0.118        0.857 +- 0.094  [5 ] TO|BT|SR|SA|SC
    0009             1a20        16        31    -15             4.79        0.516 +- 0.129        1.938 +- 0.348  [4 ] TO|BT|SR|SC
    0010             1838        43         0     43            43.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO|BT|SC|RE|AB
    0011             5c60        11         8      3             0.00        1.375 +- 0.415        0.727 +- 0.257  [6 ] EX|TO|BT|BR|SD|SC
    0012             1830        19         0     19             0.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO|BT|SC|RE
    0013             1c20        12         4      8             0.00        3.000 +- 0.866        0.333 +- 0.167  [4 ] TO|BT|BR|SC
    0014             1a28         8         8      0             0.00        1.000 +- 0.354        1.000 +- 0.354  [5 ] TO|BT|SR|SC|AB
    0015             1c28         6         9     -3             0.00        0.667 +- 0.272        1.500 +- 0.500  [5 ] TO|BT|BR|SC|AB
    0016             1428        13         1     12             0.00       13.000 +- 3.606        0.077 +- 0.077  [4 ] TO|BR|SC|AB
    0017             9c60         7         6      1             0.00        1.167 +- 0.441        0.857 +- 0.350  [6 ] EC|TO|BT|BR|SD|SC
    0018             18b0        12         0     12             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO|BT|SA|SC|RE
    .                             100000    100000       139.94/10 = 13.99  (pval:0.000 prob:1.000)  



issue : bug in pflags ab.flg makes them very inconsistent with ab.ahis (seqhis) 
------------------------------------------------------------------------------------

::

    ab.ahis
    .    all_seqhis_ana  cfo:sum  2:g4live:tds3ip   -2:g4live:tds3ip        c2        ab        ba
    .                             100000    100000       355.26/14 = 25.38  (pval:0.000 prob:1.000)
       n             iseq         a         b    a-b               c2          a/b                   b/a           [ns] label
    0000               8d     92759     92705     54             0.02        1.001 +- 0.003        0.999 +- 0.003  [2 ] TO SA
    0001               4d      5997      5980     17             0.02        1.003 +- 0.013        0.997 +- 0.013  [2 ] TO AB
    0002             7c6d       253       288    -35             2.26        0.878 +- 0.055        1.138 +- 0.067  [4 ] TO SC BT SD
    0003              86d       200       255    -55             6.65        0.784 +- 0.055        1.275 +- 0.080  [3 ] TO SC SA
    0004            4cc6d        56       173   -117            59.78        0.324 +- 0.043        3.089 +- 0.235  [5 ] TO SC BT BT AB
    0005              46d       144        61     83            33.60        2.361 +- 0.197        0.424 +- 0.054  [3 ] TO SC AB
    0006             8c6d        80        73      7             0.32        1.096 +- 0.123        0.912 +- 0.107  [4 ] TO SC BT SA
    0007           8cac6d        81         0     81            81.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO SC BT SR BT SA
    0008          8ccac6d         0        80    -80            80.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO SC BT SR BT BT SA
    0009           46cc6d        16        50    -34            17.52        0.320 +- 0.080        3.125 +- 0.442  [6 ] TO SC BT BT SC AB
    0010             4c6d        40        23     17             4.59        1.739 +- 0.275        0.575 +- 0.120  [4 ] TO SC BT AB
    0011            7cc6d         5        38    -33            25.33        0.132 +- 0.059        7.600 +- 1.233  [5 ] TO SC BT BT SD
    0012        7ccc6cc6d        28        11     17             7.41        2.545 +- 0.481        0.393 +- 0.118  [9 ] TO SC BT BT SC BT BT BT SD
    0013            8cc6d        10        24    -14             5.76        0.417 +- 0.132        2.400 +- 0.490  [5 ] TO SC BT BT SA
    0014       ccacccac6d         0        31    -31            31.00        0.000 +- 0.000        0.000 +- 0.000  [10] TO SC BT SR BT BT BT SR BT BT
    0015           4ccc6d         7        23    -16             0.00        0.304 +- 0.115        3.286 +- 0.685  [6 ] TO SC BT BT BT AB
    0016         7ccccc6d        21         3     18             0.00        7.000 +- 1.528        0.143 +- 0.082  [8 ] TO SC BT BT BT BT BT SD
    0017          466cc6d        11        11      0             0.00        1.000 +- 0.302        1.000 +- 0.302  [7 ] TO SC BT BT SC SC AB
    0018            7cb6d        10        12     -2             0.00        0.833 +- 0.264        1.200 +- 0.346  [5 ] TO SC BR BT SD
    .                             100000    100000       355.26/14 = 25.38  (pval:0.000 prob:1.000)
    ab.flg
    .       pflags_ana  cfo:sum  2:g4live:tds3ip   -2:g4live:tds3ip        c2        ab        ba
    .                             100000    100000    199766.00/17 = 11750.94  (pval:0.000 prob:1.000)
       n             iseq         a         b    a-b               c2          a/b                   b/a           [ns] label
    0000             1882     92759         0   92759         92759.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO|BT|SA|SI
    0001             1080         0     92705   -92705         92705.00        0.000 +- 0.000        0.000 +- 0.000  [2 ] TO|SA
    0002             188a      5997         0   5997          5997.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO|BT|SA|AB|SI
    0003             1008         0      5980   -5980          5980.00        0.000 +- 0.000        0.000 +- 0.000  [2 ] TO|AB

    ^^^^^^^^^^^^^^^^^^^^^ messy : zeros here points to a:OK bug which is happening with input photons only ? ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    0004             18a2       319         0    319           319.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO|BT|SA|SC|SI
    0005             18aa       314         0    314           314.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO|BT|SA|SC|AB|SI
    0006             1828         0       291   -291           291.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO|BT|SC|AB
    0007             10a0         0       258   -258           258.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO|SA|SC
    0008             5860         0       215   -215           215.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] EX|TO|BT|SD|SC
    0009             58e2       181         0    181           181.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] EX|TO|BT|SA|SD|SC|SI
    0010             9860         0       142   -142           142.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] EC|TO|BT|SD|SC
    0011             98e2       136         0    136           136.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] EC|TO|BT|SA|SD|SC|SI
    0012             18a0         0       123   -123           123.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO|BT|SA|SC
    0013             1aa2       114         0    114           114.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO|BT|SR|SA|SC|SI
    0014             1aa0         0        88    -88            88.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO|BT|SR|SA|SC
    0015             1028         0        61    -61            61.00        0.000 +- 0.000        0.000 +- 0.000  [3 ] TO|SC|AB
    0016             1a20         0        42    -42            42.00        0.000 +- 0.000        0.000 +- 0.000  [4 ] TO|BT|SR|SC
    0017             18ba        41         0     41            41.00        0.000 +- 0.000        0.000 +- 0.000  [7 ] TO|BT|SA|SC|RE|AB|SI
    0018             18b2        28         0     28             0.00        0.000 +- 0.000        0.000 +- 0.000  [6 ] TO|BT|SA|SC|RE|SI
    .                             100000    100000    199766.00/17 = 11750.94  (pval:0.000 prob:1.000)



Crazyness even apparent with a single photon::

    In [1]: a.pflags
    Out[1]: A([6274, 6274, 6274, ..., 6274, 6274, 6282], dtype=uint32)

    In [2]: b.pflags
    Out[2]: A([4224, 4104, 4224, ..., 4224, 4224, 4224], dtype=uint32)


    In [4]: hm = a.hismask

    In [5]: hm.label(6274)
    Out[5]: 'TO|BT|SA|SI'

    In [6]: hm.label(4224)
    Out[6]: 'TO|SA'


    In [9]: ht.label(a.seqhis[0])
    Out[9]: 'TO SA'

    In [10]: hm.label(a.pflags[0])
    Out[10]: 'TO|BT|SA|SI'             ## THATS AN OBVIOUS BUG : SHOULD NEVER HAVE TO|SI TOGETHER 

    In [11]: hm.label(b.pflags[0])
    Out[11]: 'TO|SA'



AHHA.  The input photons were repeated 100k times from a particular photon,
and I did not scrub the initial flags. Clearly it is necessary to do so.

oxrap/cu/generate.cu::

    628     else if(gencode == OpticksGenstep_EMITSOURCE)
    629     {
    630         // source_buffer is input only, photon_buffer output only,
    631         // photon_offset is same for both these buffers
    632         pload(p, source_buffer, photon_offset );
    633
    634         p.flags.u.x = 0u ;   // scrub any initial flags, eg when running from an input photon
    635         p.flags.u.y = 0u ;
    636         p.flags.u.z = 0u ;
    637         p.flags.u.w = 0u ;
    638
    639         s.flag = TORCH ;
    640 #ifdef WITH_REFLECT_CHEAT_DEBUG
    641         s.ureflectcheat = debug_control.w > 0u ? float(photon_id)/float(num_photon) : -1.f ;
    642 #endif
    643     }




Why didnt seq2msk checks get fired ?
---------------------------------------


::

    epsilon:ana blyth$ grep seq2msk *.py 
    evt.py:from opticks.ana.seq import SeqAna, seq2msk, SeqList
    evt.py:        jpsc = jp[np.where( seq2msk(self.seqhis[jp]) & co )]
    evt.py:        self.pflags2 = seq2msk(allseqhis)          # 16 seq nibbles OR-ed into mask 
    evt.py:            log.debug("pflags2(=seq2msk(seqhis)) and pflags  match")
    evt.py:            log.info("pflags2(=seq2msk(seqhis)) and pflags  MISMATCH    num_msk_mismatch: %d " % self.num_msk_mismatch )
    seq.py:def seq2msk_procedural(isq):
    seq.py:def seq2msk(isq):
    seq.py:        msks = seq2msk(seqs)
    tboolean.py:from opticks.ana.seq import seq2msk
    tokg4.py:from opticks.ana.seq import seq2msk
    epsilon:ana blyth$ 


::

    In [1]: a.pflags                                                                                                                                                                                          
    Out[1]: A([18498,    26,    10, ...,    10,    10,  2594], dtype=uint32)

    In [2]: a.pflags2                                                                                                                                                                                         
    Out[2]: A([2114,   26,   10, ...,   10,   10, 2594], dtype=uint64)

    In [3]: a.pflags - a.pflags2                                                                                                                                                                              
    Out[3]: A([16384,     0,     0, ...,     0,     0,     0], dtype=uint64)

    In [4]: np.count_nonzero( a.pflags - a.pflags2  )                                                                                                                                                         
    Out[4]: 3402

    In [5]: a.pflags.shape                                                                                                                                                                                    
    Out[5]: (11278,)








