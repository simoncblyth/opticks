tds3ip_pflags_inconsistency
==============================


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

    ^^^^^^^^^^^^^^^^^^^^^ messy : zeros here points to a bug ? but why with input photons only ? ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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



::

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
    Out[10]: 'TO|BT|SA|SI'

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





