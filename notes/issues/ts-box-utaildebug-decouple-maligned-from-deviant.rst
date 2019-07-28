ts-box-utaildebug-decouple-maligned-from-deviant
===================================================


Context
----------

* :doc:`tboolean-proxy-scan`



DONE : added --utaildebug which records an extra random following propagation
-------------------------------------------------------------------------------------------

Currently random mis-alignment judgement is based entirely on seqhis history of photons.  
Extending to also do a umatch, eg by recording a random at the tail 
and storing into p.weight slot.

Objective:

* decouple probable coupling between "maligned" and "deviant" issues



Conclusion
------------

* looks like utaildebug can be helpful, after getting the "tail" consumption to match 



generate.cu
--------------

::

    625     if( utaildebug )   // --utaildebug    see notes/issues/ts-box-utaildebug-decouple-maligned-from-deviant.rst
    626     {
    627         //p.weight = curand_uniform(&rng) ;
    628         p.flags.f.y = curand_uniform(&rng) ;
    629     }
    630 
    631     // breakers and maxers saved here
    632     psave(p, photon_buffer, photon_offset );


propagate.h::

    103     if (absorption_distance <= scattering_distance)
    104     {
    105         if (absorption_distance <= s.distance_to_boundary)
    106         {
    107             p.time += absorption_distance/speed ;
    108             p.position += absorption_distance*p.direction;
    109 
    110             float uniform_sample_reemit = curand_uniform(&rng);
    111             if (uniform_sample_reemit < s.material1.w)                       // .w:reemission_prob
    112             {
    113                 // no materialIndex input to reemission_lookup as both scintillators share same CDF 
    114                 // non-scintillators have zero reemission_prob
    115                 p.wavelength = reemission_lookup(curand_uniform(&rng));
    116                 p.direction = uniform_sphere(&rng);
    117                 p.polarization = normalize(cross(uniform_sphere(&rng), p.direction));
    118                 p.flags.i.x = 0 ;   // no-boundary-yet for new direction
    119 
    120                 s.flag = BULK_REEMIT ;
    121                 return CONTINUE;
    122             }
    123             else
    124             {
    125                 s.flag = BULK_ABSORB ;
    126                 return BREAK;
    127             }


modify::

    103     if (absorption_distance <= scattering_distance)
    104     {
    105         if (absorption_distance <= s.distance_to_boundary)
    106         {
    107             p.time += absorption_distance/speed ;
    108             p.position += absorption_distance*p.direction;
    109 
    110             const float& reemission_prob = s.material1.w ;
    111             float u_reemit = reemission_prob == 0.f ? 2.f : curand_uniform(&rng);  // avoid consumption at absorption when not scintillator
    112             
    113             if (u_reemit < reemission_prob)
    114             {   
    115                 // no materialIndex input to reemission_lookup as both scintillators share same CDF 
    116                 // non-scintillators have zero reemission_prob
    117                 p.wavelength = reemission_lookup(curand_uniform(&rng));
    118                 p.direction = uniform_sphere(&rng);
    119                 p.polarization = normalize(cross(uniform_sphere(&rng), p.direction));
    120                 p.flags.i.x = 0 ;   // no-boundary-yet for new direction
    121                 
    122                 s.flag = BULK_REEMIT ;
    123                 return CONTINUE;
    124             }                   
    125             else
    126             {   
    127                 s.flag = BULK_ABSORB ;
    128                 return BREAK;
    129             }
    130         }
    131         //  otherwise sail to boundary  
    132     }
    133     else








Machinery
--------------

Stomping on the weight with "--utaildebug" is inconvenient as it 
trips the ab.ox_dv, so instead stomp on identity flag: p.flags.y

::

    In [5]: np.all( ab.a.ox[:,3,1].view(np.int32) == 0 )
    Out[5]: A(True)

    In [6]: np.all( ab.b.ox[:,3,1].view(np.int32) == 0 )
    Out[6]: A(True)



Check machinery on an ok LV
-----------------------------


::

    ts box 
    ta box   # tag 1 is default 

    TAG=2 ts box --utaildebug
    TAG=2 ta box

    TAG=1,2 ta box      # ok vs ok with utaildebug
    TAG=-1,-2 ta box    # g4 vs g4 with utaildebug 
          ## perfect matches as flags are excluded from ab.ox_dv


Stomping on p.flags.y in tag 2::

    In [1]: ab.a.ox[:,3]
    Out[1]: 
    A([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       ...,
       [0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]], dtype=float32)

    In [2]: ab.b.ox[:,3]
    Out[2]: 
    A([[0.    , 0.5475, 0.    , 0.    ],
       [0.    , 0.8245, 0.    , 0.    ],
       [0.    , 0.8068, 0.    , 0.    ],
       ...,
       [0.    , 0.8001, 0.    , 0.    ],
       [0.    , 0.0093, 0.    , 0.    ],
       [0.    , 0.7396, 0.    , 0.    ]], dtype=float32)



LV 19 : which has deviations
-------------------------------

[blyth@localhost ana]$ LV=19 absmry.py
[2019-07-15 13:49:14,972] p201714 {__init__            :absmry.py :35} INFO     - base /home/blyth/local/opticks/tmp LV 19 
ABSmryTab
    LV   level   RC npho    fmal(%)  nmal                 rpost_dv.max              rpol_dv.max                ox_dv.max      solid
    19 _FATAL_ 0x05   1M      0.020    20       _FATAL_   429.2548    52   WARNING     0.0079     0   _FATAL_   429.2452     6      PMT_20inch_inner2_solid0x4cb3870
    19 _FATAL_ 0x01  10k      0.030     3       _FATAL_     0.1598     4   WARNING     0.0079     0   WARNING     0.1487     0      PMT_20inch_inner2_solid0x4cb3870


10k::

    TAG=2 ts 19 --utaildebug
    TAG=2 ta 19 
  

Comparing utail shows 3 more maligned that just seqhis comparison::

    In [1]: a.ox[:,3,1]
    Out[1]: A([0.5475, 0.8245, 0.8068, ..., 0.6484, 0.3179, 0.1052], dtype=float32)

    In [2]: b.ox[:,3,1]
    Out[2]: A([0.5475, 0.8245, 0.8068, ..., 0.6484, 0.3179, 0.1052], dtype=float32)

    In [3]: np.where( a.ox[:,3,1] != b.ox[:,3,1])
    Out[3]: (array([1872, 2084, 2908, 4074, 4860, 5477]),)

    In [4]: ab.mal.maligned
    Out[4]: array([2908, 4860, 5477])

::

    In [1]: ab.mutailed
    Out[1]: array([1872, 2084, 2908, 4074, 4860, 5477])

    In [2]: ab.dumpline(ab.mutailed)
          0   1872 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR 
          1   2084 :   :                                           TO BT AB                                           TO BT AB 
          2   2908 : * :                                  TO BT BR BR BT SA                               TO BT BR BR BR BT SA 
          3   4074 :   :                                           TO BT AB                                           TO BT AB 
          4   4860 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
          5   5477 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 



Not helping at 10k level. Pump up to 1M::

    TAG=2 ts 19 --utaildebug --generateoverride -1 

    TAG=2 ta 19


    In [3]: len(ab.misutailed)
    Out[3]: 58

    In [4]: len(ab.maligned)
    Out[4]: 20




Looks like utail mismatch for truncation and absorption::

    In [16]: np.where(np.logical_and( self.a.utail != self.b.utail, self.a.seqhis == self.b.seqhis ))[0].shape
    Out[16]: (41,)

    ## same history but mismatched utail for almost all "TO BT AB" and truncated 

    In [13]: ab.dumpline( np.where(np.logical_and( self.a.utail != self.b.utail, self.a.seqhis == self.b.seqhis ))[0] )
          0   1872 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR 
          1   2084 :   :                                           TO BT AB                                           TO BT AB 
          2   4074 :   :                                           TO BT AB                                           TO BT AB 
          3  11341 :   :                      TO BT BR BR BR BR BR BR BR BT                      TO BT BR BR BR BR BR BR BR BT 
          4  12191 :   :                                           TO BT AB                                           TO BT AB 
          5  14747 :   :                      TO BT SC BR BR BR BR BR BR BR                      TO BT SC BR BR BR BR BR BR BR 
          6  15299 :   :                                           TO BT AB                                           TO BT AB 
          7  20870 :   :                                           TO BT AB                                           TO BT AB 
          8  21502 :   :                                           TO BT AB                                           TO BT AB 
          9  25113 :   :                                        TO BT BR AB                                        TO BT BR AB 
         10  25748 :   :                                           TO BT AB                                           TO BT AB 
         11  26317 :   :                                           TO BT AB                                           TO BT AB 
         12  28413 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR 
         13  29118 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR 
         14  43525 :   :                                           TO BT AB                                           TO BT AB 
         15  45629 :   :                                     TO BT BR BR AB                                     TO BT BR BR AB 
         16  51563 :   :                                           TO BT AB                                           TO BT AB 
         17  55856 :   :                      TO BT BR BR BR BR BR BR BR BT                      TO BT BR BR BR BR BR BR BR BT 
         18  57355 :   :                                           TO BT AB                                           TO BT AB 
         19  61602 :   :                                           TO BT AB                                           TO BT AB 
         20  65189 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR 
         21  65894 :   :                                           TO BT AB                                           TO BT AB 
         22  65895 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR 
         23  68619 :   :                                           TO BT AB                                           TO BT AB 
         24  68807 :   :                                           TO BT AB                                           TO BT AB 
         25  69653 :   :                      TO BT SC BR BR BR BR BR BR BR                      TO BT SC BR BR BR BR BR BR BR 
         26  70511 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR 
         27  71280 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR 
         28  71978 :   :                                           TO BT AB                                           TO BT AB 
         29  73533 :   :                      TO BT BT SC BT BR BR BR BR BR                      TO BT BT SC BT BR BR BR BR BR 
         30  76056 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR 
         31  76427 :   :                                           TO BT AB                                           TO BT AB 
         32  77062 :   :                                           TO BT AB                                           TO BT AB 
         33  78744 :   :                                           TO BT AB                                           TO BT AB 
         34  78879 :   :                                           TO BT AB                                           TO BT AB 
         35  79117 :   :                                           TO BT AB                                           TO BT AB 
         36  81607 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR 
         37  86702 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR 
         38  86814 :   :                                           TO BT AB                                           TO BT AB 
         39  97118 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR 
         40  98796 :   :                      TO BT BR BR BR BR BR BR BR BT                      TO BT BR BR BR BR BR BR BR BT 

::

    ## manually reordered 

    [2019-07-15 15:38:34,759] p373568 {<module>            :tboolean.py:38} CRITICAL -  RC 0x05 0b101  
    [2019-07-15 15:38:34,886] p373568 {check_utaildebug    :ab.py     :194} INFO     -  u.shape:(100000, 16, 16) w.shape: (41,)   

     ua     0.6584 ub     0.6351  wa   8 wb   7     2084   2084 :   :                                           TO BT AB                                           TO BT AB   
     ua     0.4931 ub     0.9430  wa   8 wb   7     4074   4074 :   :                                           TO BT AB                                           TO BT AB   
     ua     0.3988 ub     0.2564  wa   8 wb   7    12191  12191 :   :                                           TO BT AB                                           TO BT AB   
     ua     0.4664 ub     0.0003  wa   8 wb   7    15299  15299 :   :                                           TO BT AB                                           TO BT AB   
     ua     0.5980 ub     0.7003  wa   8 wb   7    20870  20870 :   :                                           TO BT AB                                           TO BT AB   
     ua     0.1625 ub     0.9363  wa   8 wb   7    21502  21502 :   :                                           TO BT AB                                           TO BT AB   
     ua     0.1632 ub     0.0309  wa   8 wb   7    25748  25748 :   :                                           TO BT AB                                           TO BT AB   
     ua     0.2556 ub     0.4249  wa   8 wb   7    26317  26317 :   :                                           TO BT AB                                           TO BT AB   
     ua     0.7599 ub     0.0771  wa   8 wb   7    43525  43525 :   :                                           TO BT AB                                           TO BT AB   
     ua     0.7590 ub     0.4485  wa   8 wb   7    51563  51563 :   :                                           TO BT AB                                           TO BT AB   
     ua     0.0235 ub     0.6259  wa   8 wb   7    57355  57355 :   :                                           TO BT AB                                           TO BT AB   
     ua     0.9717 ub     0.9801  wa   8 wb   7    61602  61602 :   :                                           TO BT AB                                           TO BT AB   
     ua     0.0247 ub     0.6257  wa   8 wb   7    65894  65894 :   :                                           TO BT AB                                           TO BT AB   
     ua     0.3433 ub     0.5848  wa   8 wb   7    68619  68619 :   :                                           TO BT AB                                           TO BT AB   
     ua     0.3624 ub     0.6515  wa   8 wb   7    68807  68807 :   :                                           TO BT AB                                           TO BT AB   
     ua     0.4012 ub     0.1663  wa   8 wb   7    71978  71978 :   :                                           TO BT AB                                           TO BT AB   
     ua     0.1526 ub     0.1197  wa   8 wb   7    76427  76427 :   :                                           TO BT AB                                           TO BT AB   
     ua     0.8587 ub     0.1138  wa   8 wb   7    77062  77062 :   :                                           TO BT AB                                           TO BT AB   
     ua     0.8890 ub     0.6831  wa   8 wb   7    78744  78744 :   :                                           TO BT AB                                           TO BT AB   
     ua     0.6581 ub     0.6789  wa   8 wb   7    78879  78879 :   :                                           TO BT AB                                           TO BT AB   
     ua     0.4127 ub     0.8073  wa   8 wb   7    79117  79117 :   :                                           TO BT AB                                           TO BT AB   
     ua     0.7031 ub     0.0881  wa   8 wb   7    86814  86814 :   :                                           TO BT AB                                           TO BT AB   

     ua     0.7285 ub     0.5340  wa  12 wb  11    25113  25113 :   :                                        TO BT BR AB                                        TO BT BR AB   

     ua     0.1042 ub     0.4516  wa  16 wb  15    45629  45629 :   :                                     TO BT BR BR AB                                     TO BT BR BR AB   

     ua     0.7897 ub     0.2034  wa  36 wb  45    29118  29118 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     ua     0.8095 ub     0.1871  wa  36 wb  45     1872   1872 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     ua     0.7844 ub     0.0212  wa  36 wb  45    28413  28413 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     ua     0.3840 ub     0.7007  wa  36 wb  45    65189  65189 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     ua     0.8852 ub     0.4157  wa  36 wb  45    65895  65895 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     ua     0.4942 ub     0.8756  wa  36 wb  45    76056  76056 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     ua     0.5162 ub     0.1054  wa  36 wb  45    71280  71280 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     ua     0.0474 ub     0.5089  wa  36 wb  45    81607  81607 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     ua     0.3869 ub     0.8174  wa  36 wb  45    86702  86702 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     ua     0.6177 ub     0.1149  wa  36 wb  45    97118  97118 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   

     ua     0.1619 ub     0.7004  wa  36 wb  48    70511  70511 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   

     ua     0.2997 ub     0.7499  wa  36 wb  41    11341  11341 :   :                      TO BT BR BR BR BR BR BR BR BT                      TO BT BR BR BR BR BR BR BR BT   
     ua     0.8077 ub     0.1347  wa  36 wb  41    55856  55856 :   :                      TO BT BR BR BR BR BR BR BR BT                      TO BT BR BR BR BR BR BR BR BT   
     ua     0.6535 ub     0.8339  wa  36 wb  41    98796  98796 :   :                      TO BT BR BR BR BR BR BR BR BT                      TO BT BR BR BR BR BR BR BR BT   

     ua     0.6346 ub     0.8817  wa  40 wb  53    14747  14747 :   :                      TO BT SC BR BR BR BR BR BR BR                      TO BT SC BR BR BR BR BR BR BR   
     ua     0.4836 ub     0.3343  wa  40 wb  49    69653  69653 :   :                      TO BT SC BR BR BR BR BR BR BR                      TO BT SC BR BR BR BR BR BR BR   

     ua     0.7189 ub     0.0817  wa  50 wb  63    73533  73533 :   :                      TO BT BT SC BT BR BR BR BR BR                      TO BT BT SC BT BR BR BR BR BR   



* all ending with "AB" : Opticks (A) consumes 1 more than G4 (B)

  * the extra was u_reemit which was being done even when reemission_prob for material was zero  

* all the G4 truncated consuming more (5~13), some variations by 3/4 even within same history 




::

    In [1]: ab.his
    Out[1]: 
    ab.his
    .                seqhis_ana  2:tboolean-proxy-19:tboolean-proxy-19   -2:tboolean-proxy-19:tboolean-proxy-19        c2        ab        ba 
    .                             100000    100000         0.01/8 =  0.00  (pval:1.000 prob:0.000)  
    0000             8ccd     86046     86046      0             0.00        1.000 +- 0.003        1.000 +- 0.003  [4 ] TO BT BT SA
    0001              8bd      7841      7841      0             0.00        1.000 +- 0.011        1.000 +- 0.011  [3 ] TO BR SA
    0002            8cbcd      4991      4990      1             0.00        1.000 +- 0.014        1.000 +- 0.014  [5 ] TO BT BR BT SA
    0003           8cbbcd       723       722      1             0.00        1.001 +- 0.037        0.999 +- 0.037  [6 ] TO BT BR BR BT SA
    0004         8cbbbbcd       104       104      0             0.00        1.000 +- 0.098        1.000 +- 0.098  [8 ] TO BT BR BR BR BR BT SA
    0005          8cbbbcd        80        81     -1             0.01        0.988 +- 0.110        1.012 +- 0.113  [7 ] TO BT BR BR BR BT SA
    0006              86d        57        57      0             0.00        1.000 +- 0.132        1.000 +- 0.132  [3 ] TO SC SA
    0007            86ccd        51        51      0             0.00        1.000 +- 0.140        1.000 +- 0.140  [5 ] TO BT BT SC SA
     0008              4cd        22        22      0             0.00        1.000 +- 0.213        1.000 +- 0.213  [3 ] TO BT AB
    0009            8cccd        17         0     17             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO BT BT BT SA
     0010       bbbbbbbbcd        11        12     -1             0.00        0.917 +- 0.276        1.091 +- 0.315  [10] TO BT BR BR BR BR BR BR BR BR
    0011           8cb6cd         8         8      0             0.00        1.000 +- 0.354        1.000 +- 0.354  [6 ] TO BT SC BR BT SA
    0012            8c6cd         8         8      0             0.00        1.000 +- 0.354        1.000 +- 0.354  [5 ] TO BT SC BT SA
     0013       8cbbbbbbcd         5         5      0             0.00        1.000 +- 0.447        1.000 +- 0.447  [10] TO BT BR BR BR BR BR BR BT SA
     0014       cbbbbbbbcd         3         3      0             0.00        1.000 +- 0.577        1.000 +- 0.577  [10] TO BT BR BR BR BR BR BR BR BT
    0015            8cc6d         3         3      0             0.00        1.000 +- 0.577        1.000 +- 0.577  [5 ] TO SC BT BT SA
    0016          8cc6ccd         3         3      0             0.00        1.000 +- 0.577        1.000 +- 0.577  [7 ] TO BT BT SC BT BT SA
    0017             86bd         3         3      0             0.00        1.000 +- 0.577        1.000 +- 0.577  [4 ] TO BR SC SA
    0018           86cbcd         2         2      0             0.00        1.000 +- 0.707        1.000 +- 0.707  [6 ] TO BT BR BT SC SA
     0019       8cbbbc6ccd         2         2      0             0.00        1.000 +- 0.707        1.000 +- 0.707  [10] TO BT BT SC BT BR BR BR BT SA
    .                             100000    100000         0.01/8 =  0.00  (pval:1.000 prob:0.000)  

    In [2]: 22+12+5+3+2
    Out[2]: 44




Off by one in the sequence::

    In [6]: u = np.load("/tmp/blyth/opticks/TRngBufTest.npy").astype(np.float32)

    In [8]: u[2084]
    Out[8]: 
    array([[0.9537, 0.0564, 0.4223, 0.0844, 0.613 , 0.5363, 0.9999, 0.6351, 0.6584, 0.2606, 0.8613, 0.7033, 0.8223, 0.6353, 0.388 , 0.2703],
           [0.4434, 0.8683, 0.4154, 0.7569, 0.0229, 0.7002, 0.8288, 0.6337, 0.9668, 0.4033, 0.6487, 0.5053, 0.7157, 0.3847, 0.269 , 0.3033],
           [0.9397, 0.6064, 0.0327, 0.3712, 0.6245, 0.3466, 0.5606, 0.5509, 0.3882, 0.1086, 0.768 , 0.7768, 0.8073, 0.9359, 0.836 , 0.9718],
           [0.0275, 0.1327, 0.6782, 0.2846, 0.9909, 0.1524, 0.2576, 0.7536, 0.137 , 0.8297, 0.5487, 0.4995, 0.9066, 0.3126, 0.7749, 0.8859],
           [0.256 , 0.1372, 0.0653, 0.5853, 0.5436, 0.6742, 0.02  , 0.3734, 0.7504, 0.6284, 0.0362, 0.3037, 0.6273, 0.105 , 0.8729, 0.9207],
            ...

    In [11]: u[2084].shape
    Out[11]: (16, 16)

    In [9]: np.where( u[2084].astype(np.float32) == a.utail[2084] )
    Out[9]: (array([0]), array([8]))

    In [10]: a.utail[2084]
    Out[10]: 0.6583896

    In [12]: np.where( u[2084].astype(np.float32) == b.utail[2084] )
    Out[12]: (array([0]), array([7]))






