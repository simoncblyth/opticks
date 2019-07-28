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


After avoiding u_reemit consumption on AB for non-scintillator

::

    def _get_misutailed(self):
        return np.where(self.a.utail != self.b.utail)[0]

::

    In [2]:  ab.misutailed
    Out[2]: array([1872, 2908, 4860, 5477])

    In [3]: ab.dumpline(ab.misutailed)
          0   1872 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR 
          1   2908 : * :                                  TO BT BR BR BT SA                               TO BT BR BR BR BT SA 
          2   4860 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
          3   5477 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 









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




After avoid u_reemit consumption when reemission_prob is zero (not a scintillator)
-------------------------------------------------------------------------------------


10k::


    OpticksProfile=ERROR NEmitPhotonsNPY=ERROR TAG=2 ts 19 --dbgemit 

         # initially was taking ages to generate photons
         # due to bad ranges umin/umax/vmin/vmax  were all equal at 0.45


    TAG=2 ts 19 --utaildebug   ## 100k
    TAG=2 ta 19 
        # results incorporated above, show expected removal of "TO BT AB" from the misutailed
        # leaving truncated + history maligned 


1M::

    TAG=2 ts 19 --utaildebug --generateoverride -1      ## 1M 

    ta 19 --tag 2 --msli :1M

    TRngBuf_NI=1000000 TRngBufTest     ## create 2GB array of randoms



ta 19 --tag 2 --msli :1M::

    [2019-07-28 12:29:42,770] p52724 {<module>            :tboolean.py:38} CRITICAL -  RC 0x05 0b101 
    [2019-07-28 12:29:42,792] p52724 {check_utaildebug    :ab.py     :204} INFO     - utail mismatch but seqhis matched u.shape:(1000000, 16, 16) w.shape: (126,) 
     i     0 p    1872 ua     0.8095 ub     0.1871  wa  36 wb  45 wb-wa   9 :    1872   1872 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i     1 p   11341 ua     0.2997 ub     0.7499  wa  36 wb  41 wb-wa   5 :   11341  11341 :   :                      TO BT BR BR BR BR BR BR BR BT                      TO BT BR BR BR BR BR BR BR BT   
     i     2 p   14747 ua     0.6346 ub     0.8817  wa  40 wb  53 wb-wa  13 :   14747  14747 :   :                      TO BT SC BR BR BR BR BR BR BR                      TO BT SC BR BR BR BR BR BR BR   
     i     3 p   28413 ua     0.7844 ub     0.0212  wa  36 wb  45 wb-wa   9 :   28413  28413 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i     4 p   29118 ua     0.7897 ub     0.2034  wa  36 wb  45 wb-wa   9 :   29118  29118 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i     5 p   55856 ua     0.8077 ub     0.1347  wa  36 wb  41 wb-wa   5 :   55856  55856 :   :                      TO BT BR BR BR BR BR BR BR BT                      TO BT BR BR BR BR BR BR BR BT   
     i     6 p   65189 ua     0.3840 ub     0.7007  wa  36 wb  45 wb-wa   9 :   65189  65189 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i     7 p   65895 ua     0.8852 ub     0.4157  wa  36 wb  45 wb-wa   9 :   65895  65895 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i     8 p   69653 ua     0.4836 ub     0.3343  wa  40 wb  49 wb-wa   9 :   69653  69653 :   :                      TO BT SC BR BR BR BR BR BR BR                      TO BT SC BR BR BR BR BR BR BR   
     ...
     the count 126-2 (2 scatterers) in 1M corresponds roughly to the total of truncated as gleaned from ab.ahis[:100] by eye  
     so looks like all truncated are utail discrepant with G4 always consuming 5-18 more randoms

     i    34 p  258609 ua     0.2699 ub     0.5301  wa  35 wb  30 wb-wa  -5 :  258609 258609 :   :                                  TO BT BR BT SC SA                                  TO BT BR BT SC SA   
     i   114 p  892900 ua     0.1056 ub     0.3332  wa  35 wb  25 wb-wa -10 :  892900 892900 :   :                                  TO BT BR BT SC SA                                  TO BT BR BT SC SA   

     rayleigh scattering consumes 5u at each turn of its loop, so -5 / -10 may be explained by the loop termination "edge" 


    In [1]: 


TODO: masked running on a truncated photon eg 1872

::

     TAG=2 ts 19 --utaildebug --dbgseqhis 0xbbbbbbbbcd --generateoverride -1
     # must specify the tag via envvar 
 


Records continue to be collected beyond the truncate::

    2019-07-28 15:02:51.585 INFO  [70133] [CRec::dump@172]  nstp 19
    ( 0)  TO/BT     FrT                       PRE_SAVE POST_SAVE STEP_START 
    [   0](Stp ;opticalphoton stepNum   19(tk ;opticalphoton tid 9207 pid 0 nm    380 mm  ori[   71.714 -14.972-746.900]  pos[ -116.327  24.2871493.900]  )
      pre                  box_pv0_          Vacuum          noProc           Undefined pos[      0.000     0.000     0.000]  dir[   -0.000  -0.000   1.000]  pol[    0.000  -1.000   0.000]  ns  0.000 nm 380.000 mm/ns 299.792
     post                union_pv0_   GlassSchottF2  Transportation        GeomBoundary pos[      0.000     0.000   731.501]  dir[   -0.396   0.083   0.915]  pol[   -0.398  -0.913  -0.090]  ns  2.440 nm 380.000 mm/ns 165.028
     )
    ( 1)  BT/BR     FrR                                  POST_SAVE MAT_SWAP 
    [   1](Stp ;opticalphoton stepNum   19(tk ;opticalphoton tid 9207 pid 0 nm    380 mm  ori[   71.714 -14.972-746.900]  pos[ -116.327  24.2871493.900]  )
      pre                union_pv0_   GlassSchottF2  Transportation        GeomBoundary pos[      0.000     0.000   731.501]  dir[   -0.396   0.083   0.915]  pol[   -0.398  -0.913  -0.090]  ns  2.440 nm 380.000 mm/ns 165.028
     post                  box_pv0_          Vacuum  Transportation        GeomBoundary pos[    -78.892    16.471   913.905]  dir[   -0.396   0.083  -0.915]  pol[   -0.116  -0.992  -0.039]  ns  3.648 nm 380.000 mm/ns 165.028
     )

    ...

    (14)  BR/NA     STS                                           POST_SKIP 
    [  14](Stp ;opticalphoton stepNum   19(tk ;opticalphoton tid 9207 pid 0 nm    380 mm  ori[   71.714 -14.972-746.900]  pos[ -116.327  24.2871493.900]  )
      pre                  box_pv0_          Vacuum  Transportation        GeomBoundary pos[   -164.081    34.256   748.255]  dir[   -0.756   0.158   0.635]  pol[   -0.203  -0.979   0.001]  ns  9.794 nm 380.000 mm/ns 165.028
     post                union_pv0_   GlassSchottF2  Transportation        GeomBoundary pos[   -164.081    34.256   748.255]  dir[   -0.756   0.158   0.635]  pol[   -0.203  -0.979   0.001]  ns  9.794 nm 380.000 mm/ns 165.028
     )
    (15)  NA/BR     TIR   POST_SAVE POST_DONE MAT_SWAP RECORD_TRUNCATE BOUNCE_TRUNCATE 
    [  15](Stp ;opticalphoton stepNum   19(tk ;opticalphoton tid 9207 pid 0 nm    380 mm  ori[   71.714 -14.972-746.900]  pos[ -116.327  24.2871493.900]  )
      pre                union_pv0_   GlassSchottF2  Transportation        GeomBoundary pos[   -164.081    34.256   748.255]  dir[   -0.756   0.158   0.635]  pol[   -0.203  -0.979   0.001]  ns  9.794 nm 380.000 mm/ns 165.028
     post                  box_pv0_          Vacuum  Transportation        GeomBoundary pos[   -307.540    64.208   868.652]  dir[    0.181  -0.038   0.983]  pol[    0.206   0.979  -0.000]  ns 10.943 nm 380.000 mm/ns 165.028
     )
    (16)    /       STS                                                     
    [  16](Stp ;opticalphoton stepNum   19(tk ;opticalphoton tid 9207 pid 0 nm    380 mm  ori[   71.714 -14.972-746.900]  pos[ -116.327  24.2871493.900]  )
      pre                  box_pv0_          Vacuum  Transportation        GeomBoundary pos[   -307.540    64.208   868.652]  dir[    0.181  -0.038   0.983]  pol[    0.206   0.979  -0.000]  ns 10.943 nm 380.000 mm/ns 165.028
     post                union_pv0_   GlassSchottF2  Transportation        GeomBoundary pos[   -307.540    64.208   868.652]  dir[    0.181  -0.038   0.983]  pol[    0.206   0.979  -0.000]  ns 10.943 nm 380.000 mm/ns 165.028
     )
    (17)    /       FrT                                                     
    [  17](Stp ;opticalphoton stepNum   19(tk ;opticalphoton tid 9207 pid 0 nm    380 mm  ori[   71.714 -14.972-746.900]  pos[ -116.327  24.2871493.900]  )
      pre                union_pv0_   GlassSchottF2  Transportation        GeomBoundary pos[   -307.540    64.208   868.652]  dir[    0.181  -0.038   0.983]  pol[    0.206   0.979  -0.000]  ns 10.943 nm 380.000 mm/ns 165.028
     post                  box_pv0_          Vacuum  Transportation        GeomBoundary pos[   -299.214    62.469   913.905]  dir[    0.300  -0.063   0.952]  pol[    0.203   0.979   0.000]  ns 11.222 nm 380.000 mm/ns 299.792
     )
    (18)    /       Abs                                                     
    [  18](Stp ;opticalphoton stepNum   19(tk ;opticalphoton tid 9207 pid 0 nm    380 mm  ori[   71.714 -14.972-746.900]  pos[ -116.327  24.2871493.900]  )
      pre                  box_pv0_          Vacuum  Transportation        GeomBoundary pos[   -299.214    62.469   913.905]  dir[    0.300  -0.063   0.952]  pol[    0.203   0.979   0.000]  ns 11.222 nm 380.000 mm/ns 299.792
     post               UNIVERSE_PV            Rock  Transportation        GeomBoundary pos[   -116.327    24.287  1493.900]  dir[    0.300  -0.063   0.952]  pol[    0.203   0.979   0.000]  ns 13.255 nm 380.000 mm/ns 299.792
     )
    2019-07-28 15:02:51.586 INFO  [70133] [CRec::dump@176]  npoi 0



Hmm need to be using "--recpoi --recpoialign" for the tail consumption to match ?
---------------------------------------------------------------------------------------

* nope "--recpoalign" prevents truncation to make recstp and recpoi truncate at same place, 
  which is exactly what you do not want for same tail consumption ... just need "--recpoi" ?

* :doc:`cfg4-recpoi-recstp-insidious-difference`



With this things look much the same::

     TAG=2 ts 19 --utaildebug --dbgseqhis 0xbbbbbbbbcd --generateoverride -1 --recpoi --recpoialign 

     # must specify the tag via envvar, as tboolean-proxy already  setting it 

     ta 19 --tag 2 --msli :1M


1M with "--recpoi" and without "--recpoialign" 
---------------------------------------------------


::

     TAG=2 ts 19 --utaildebug --dbgseqhis 0xbbbbbbbbcd --generateoverride -1 --recpoi 

     ta 19 --tag 2 --msli :1M




Using "--recpoi" brings truncation tail consumption closer but still 3/4/5/8 more consumed on G4 side::

    [2019-07-28 16:53:47,356] p109324 {<module>            :tboolean.py:38} CRITICAL -  RC 0x05 0b101 
    [2019-07-28 16:53:47,378] p109324 {check_utaildebug    :ab.py     :204} INFO     - utail mismatch but seqhis matched u.shape:(1000000, 16, 16) w.shape: (127,) 
     i     0 p    1872 ua     0.8095 ub     0.8632  wa  36 wb  40 wb-wa   4 :    1872   1872 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i     1 p   11341 ua     0.2997 ub     0.7499  wa  36 wb  41 wb-wa   5 :   11341  11341 :   :                      TO BT BR BR BR BR BR BR BR BT                      TO BT BR BR BR BR BR BR BR BT   
     i     2 p   14747 ua     0.6346 ub     0.3808  wa  40 wb  43 wb-wa   3 :   14747  14747 :   :                      TO BT SC BR BR BR BR BR BR BR                      TO BT SC BR BR BR BR BR BR BR   
     i     3 p   28413 ua     0.7844 ub     0.1908  wa  36 wb  40 wb-wa   4 :   28413  28413 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i     4 p   29118 ua     0.7897 ub     0.1516  wa  36 wb  40 wb-wa   4 :   29118  29118 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i     5 p   55856 ua     0.8077 ub     0.1347  wa  36 wb  41 wb-wa   5 :   55856  55856 :   :                      TO BT BR BR BR BR BR BR BR BT                      TO BT BR BR BR BR BR BR BR BT   
     i     6 p   65189 ua     0.3840 ub     0.8216  wa  36 wb  40 wb-wa   4 :   65189  65189 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i     7 p   65895 ua     0.8852 ub     0.1081  wa  36 wb  40 wb-wa   4 :   65895  65895 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i     8 p   69653 ua     0.4836 ub     0.8836  wa  40 wb  44 wb-wa   4 :   69653  69653 :   :                      TO BT SC BR BR BR BR BR BR BR                      TO BT SC BR BR BR BR BR BR BR   
     i     9 p   70511 ua     0.1619 ub     0.6413  wa  36 wb  40 wb-wa   4 :   70511  70511 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    10 p   71280 ua     0.5162 ub     0.5134  wa  36 wb  40 wb-wa   4 :   71280  71280 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    11 p   73533 ua     0.7189 ub     0.8841  wa  50 wb  53 wb-wa   3 :   73533  73533 :   :                      TO BT BT SC BT BR BR BR BR BR                      TO BT BT SC BT BR BR BR BR BR   
     i    12 p   76056 ua     0.4942 ub     0.3190  wa  36 wb  40 wb-wa   4 :   76056  76056 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    13 p   81607 ua     0.0474 ub     0.4509  wa  36 wb  40 wb-wa   4 :   81607  81607 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    14 p   86702 ua     0.3869 ub     0.1979  wa  36 wb  40 wb-wa   4 :   86702  86702 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    15 p   97118 ua     0.6177 ub     0.5148  wa  36 wb  40 wb-wa   4 :   97118  97118 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    16 p   98796 ua     0.6535 ub     0.8339  wa  36 wb  41 wb-wa   5 :   98796  98796 :   :                      TO BT BR BR BR BR BR BR BR BT                      TO BT BR BR BR BR BR BR BR BT   
     i    17 p  107799 ua     0.6279 ub     0.9544  wa  40 wb  43 wb-wa   3 :  107799 107799 :   :                      TO BT SC BR BR BR BR BR BR BR                      TO BT SC BR BR BR BR BR BR BR   
     i    18 p  133597 ua     0.3362 ub     0.2959  wa  36 wb  40 wb-wa   4 :  133597 133597 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    19 p  134001 ua     0.3896 ub     0.5885  wa  36 wb  39 wb-wa   3 :  134001 134001 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    20 p  161958 ua     0.1926 ub     0.5091  wa  36 wb  41 wb-wa   5 :  161958 161958 :   :                      TO BT BR BR BR BR BR BR BR BT                      TO BT BR BR BR BR BR BR BR BT   
     i    21 p  173028 ua     0.1073 ub     0.7930  wa  36 wb  40 wb-wa   4 :  173028 173028 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    22 p  181493 ua     0.7031 ub     0.8160  wa  36 wb  40 wb-wa   4 :  181493 181493 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    23 p  188722 ua     0.6362 ub     0.6660  wa  36 wb  40 wb-wa   4 :  188722 188722 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    24 p  193421 ua     0.2367 ub     0.3336  wa  36 wb  40 wb-wa   4 :  193421 193421 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    25 p  214880 ua     0.7284 ub     0.4032  wa  45 wb  53 wb-wa   8 :  214880 214880 :   :                      TO BT BT SC BT BT BT BR BR BT                      TO BT BT SC BT BT BT BR BR BT   
     i    26 p  224442 ua     0.1189 ub     0.2518  wa  36 wb  40 wb-wa   4 :  224442 224442 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    27 p  225620 ua     0.2323 ub     0.1610  wa  45 wb  48 wb-wa   3 :  225620 225620 :   :                      TO BT BT SC BT BR BR BR BR BR                      TO BT BT SC BT BR BR BR BR BR   
     i    28 p  228255 ua     0.0643 ub     0.3217  wa  45 wb  48 wb-wa   3 :  228255 228255 :   :                      TO BT SC BR BR BR BR BR BR BR                      TO BT SC BR BR BR BR BR BR BR   
     i    29 p  230971 ua     0.9279 ub     0.7837  wa  36 wb  40 wb-wa   4 :  230971 230971 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    30 p  238905 ua     0.0739 ub     0.9271  wa  36 wb  40 wb-wa   4 :  238905 238905 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    31 p  239936 ua     0.0040 ub     0.0103  wa  36 wb  40 wb-wa   4 :  239936 239936 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    32 p  245751 ua     0.8306 ub     0.8158  wa  40 wb  44 wb-wa   4 :  245751 245751 :   :                      TO BT SC BR BR BR BR BR BR BR                      TO BT SC BR BR BR BR BR BR BR   
     i    33 p  249877 ua     0.1762 ub     0.6996  wa  36 wb  39 wb-wa   3 :  249877 249877 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    34 p  258609 ua     0.2699 ub     0.5301  wa  35 wb  30 wb-wa  -5 :  258609 258609 :   :                                  TO BT BR BT SC SA                                  TO BT BR BT SC SA   
     i    35 p  273479 ua     0.0259 ub     0.6191  wa  36 wb  40 wb-wa   4 :  273479 273479 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    36 p  277020 ua     0.3378 ub     0.9429  wa  36 wb  41 wb-wa   5 :  277020 277020 :   :                      TO BT BR BR BR BR BR BR BR BT                      TO BT BR BR BR BR BR BR BR BT   
     i    37 p  279901 ua     0.0662 ub     0.4537  wa  50 wb  53 wb-wa   3 :  279901 279901 :   :                      TO BT BT SC BT BR BR BR BR BR                      TO BT BT SC BT BR BR BR BR BR   
     i    38 p  285218 ua     0.9254 ub     0.3923  wa  36 wb  40 wb-wa   4 :  285218 285218 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    39 p  289122 ua     0.9409 ub     0.2575  wa  36 wb  41 wb-wa   5 :  289122 289122 :   :                      TO BT BR BR BR BR BR BR BR BT                      TO BT BR BR BR BR BR BR BR BT   
     i    40 p  290370 ua     0.8984 ub     0.0157  wa  36 wb  40 wb-wa   4 :  290370 290370 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    41 p  298272 ua     0.1088 ub     0.3684  wa  36 wb  41 wb-wa   5 :  298272 298272 :   :                      TO BT BR BR BR BR BR BR BR BT                      TO BT BR BR BR BR BR BR BR BT   
     i    42 p  317426 ua     0.6457 ub     0.9483  wa  36 wb  40 wb-wa   4 :  317426 317426 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    43 p  318849 ua     0.7922 ub     0.4005  wa  36 wb  40 wb-wa   4 :  318849 318849 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    44 p  319783 ua     0.9959 ub     0.1055  wa  36 wb  41 wb-wa   5 :  319783 319783 :   :                      TO BT BR BR BR BR BR BR BR BT                      TO BT BR BR BR BR BR BR BR BT   
     i    45 p  351371 ua     0.0695 ub     0.8510  wa  36 wb  40 wb-wa   4 :  351371 351371 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    46 p  352477 ua     0.2692 ub     0.3851  wa  36 wb  40 wb-wa   4 :  352477 352477 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    47 p  355996 ua     0.8041 ub     0.8757  wa  13 wb  12 wb-wa  -1 :  355996 355996 :   :                                        TO BT BT SA                                        TO BT BT SA   
    

* looks like an off-by-1

Try just using bounce_max for the point limit to correspond to generate.cu
-------------------------------------------------------------------------------

::

     86 unsigned CG4Ctx::point_limit() const
     87 {
     88     assert( _ok_event_init );
     89     //return ( _steps_per_photon > _bounce_max ? _steps_per_photon : _bounce_max ) ;
     90     return _bounce_max  ;
     91 }

::

     TAG=2 ts 19 --utaildebug --dbgseqhis 0xbbbbbbbbcd --generateoverride -1 --recpoi 

     ta 19 --tag 2 --msli :1M


With that are down to 3 with utail mismatch but seqhis matched::

    [2019-07-28 17:28:25,684] p118051 {<module>            :tboolean.py:38} CRITICAL -  RC 0x05 0b101 
    [2019-07-28 17:28:25,699] p118051 {check_utaildebug    :ab.py     :204} INFO     - utail mismatch but seqhis matched u.shape:(1000000, 16, 16) w.shape: (3,) 
     i     0 p  258609 ua     0.2699 ub     0.5301  wa  35 wb  30 wd  -5 :  258609 258609 :   :                                  TO BT BR BT SC SA                                  TO BT BR BT SC SA   
     i     1 p  635008 ua     0.5399 ub     0.5543  wa  13 wb  12 wd  -1 :  635008 635008 :   :                                        TO BT BT SA                                        TO BT BT SA   
     i     2 p  892900 ua     0.1056 ub     0.3332  wa  35 wb  25 wd -10 :  892900 892900 :   :                                  TO BT BR BT SC SA                                  TO BT BR BT SC SA   

    In [1]: 


Argh, but the truncated getting a badflag seqhis 0::

    ab.mal
    aligned   999712/1000000 : 0.9997 : 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 
    maligned     288/1000000 : 0.0003 : 1872,2908,4860,5477,11341,12338,14747,17891,18117,28413,28709,29118,32764,37671,43675,45874,46032,55856,60178,63381,65189,65895,69653,70511,71280 
    slice(0, 25, None)
          0   1872 : * :                      TO BT BR BR BR BR BR BR BR BR                                                ?0? 
          1   2908 : * :                                  TO BT BR BR BT SA                               TO BT BR BR BR BT SA 
          2   4860 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
          3   5477 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
          4  11341 : * :                      TO BT BR BR BR BR BR BR BR BT                                                ?0? 
          5  12338 : * :                                     TO BT BR BT SA                                                ?0? 
          6  14747 : * :                      TO BT SC BR BR BR BR BR BR BR                                                ?0? 
          7  17891 : * :                                     TO BT BT BT SA                                     TO BT BT BR SA 
          8  18117 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
          9  28413 : * :                      TO BT BR BR BR BR BR BR BR BR                                                ?0? 
         10  28709 : * :                                     TO BT BT BT SA                                     TO BT BT BR SA 
         11  29118 : * :                      TO BT BR BR BR BR BR BR BR BR                                                ?0? 
         12  32764 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
         13  37671 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
         14  43675 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
         15  45874 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
         16  46032 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
         17  55856 : * :                      TO BT BR BR BR BR BR BR BR BT                                                ?0? 
         18  60178 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
         19  63381 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
         20  65189 : * :                      TO BT BR BR BR BR BR BR BR BR                                                ?0? 
         21  65895 : * :                      TO BT BR BR BR BR BR BR BR BR                                                ?0? 
         22  69653 : * :                      TO BT SC BR BR BR BR BR BR BR                                                ?0? 
         23  70511 : * :                      TO BT BR BR BR BR BR BR BR BR                                                ?0? 
         24  71280 : * :                      TO BT BR BR BR BR BR BR BR BR                                                ?0? 
    .
    [2019-07-28 17:28:25,671] p118051 {code                :seq.py    :171} WARNING  - SeqType.code check [?0?] bad 1 
    [2019-07-28 17:28:25,677] p118051 {code                :seq.py    :171} WARNING  - SeqType.code check [?0?] bad 1 
         

::

    In [1]: a.seqhis
    Out[1]: A([36045, 36045,  2237, ..., 36045, 36045, 36045], dtype=uint64)

    In [2]: a.seqhis[1872]
    Out[2]: 806308527053

    In [3]: "%0x" % a.seqhis[1872]
    Out[3]: 'bbbbbbbbcd'

    In [4]: "%0x" % b.seqhis[1872]
    Out[4]: '0'

    In [5]:  b.seqhis[1872]
    Out[5]: 0

    In [6]: b.seqhis
    Out[6]: A([36045, 36045,  2237, ..., 36045, 36045, 36045], dtype=uint64)



::
     TAG=2 ts 19 --utaildebug --dbgseqhis 0x0 --dbgseqmat 0x1 --generateoverride -1 --recpoi 

     ta 19 --tag 2 --msli :1M



Back to the old point_limit avoids the seqhis zeros, but have the tail consumption mismatch::

     86 unsigned CG4Ctx::point_limit() const
     87 {
     88     assert( _ok_event_init );
     89     return ( _steps_per_photon > _bounce_max ? _steps_per_photon : _bounce_max ) ;
     90     //return _bounce_max  ;
     91 }


* TODO: find the cause of the seqhis 0, suspect hard_truncation  




Hmm confusing split between collection and writing. 
-----------------------------------------------------

* suspect conflation between these causing the truncation issues, the 
  "done" on the writing side doesnt kill tracks : so cannot control utaildebug consumption  


--recpoi --reccf shows truncation difference between the modes : for ~30/1M
------------------------------------------------------------------------------------

::

    TAG=2 CRecorder=ERROR ts 19 --utaildebug --generateoverride -1 --recpoi --reccf 

::

    2019-07-28 20:27:46.703 INFO  [161752] [CInputPhotonSource::GeneratePrimaryVertex@184]  num_photons 10000 gpv_count 98 event_gencode 4096 : TORCH
    2019-07-28 20:27:48.026 INFO  [161752] [CInputPhotonSource::GeneratePrimaryVertex@184]  num_photons 10000 gpv_count 99 event_gencode 4096 : TORCH
    2019-07-28 20:27:48.487 ERROR [161752] [CRecorder::postTrackWritePoints@335]  done and not last  i 9 numPoi 11
    2019-07-28 20:27:48.487 INFO  [161752] [CRecorder::compareModes@194]  record_id 996623 event_id 99 track_id 6623 photon_id 6623 parent_id -1 primary_id -2 reemtrack 0
    2019-07-28 20:27:48.487 INFO  [161752] [CRecorder::compareModes@195] ps:CPhoton slot_constrained 9 seqhis           cbbbbc6ccd seqmat           4111114414 is_flag_done N is_done Y
    2019-07-28 20:27:48.487 INFO  [161752] [CRecorder::compareModes@196] pp:CPhoton slot_constrained 9 seqhis           8bbbbc6ccd seqmat           3111114414 is_flag_done Y is_done Y
    2019-07-28 20:27:49.248 ERROR [161752] [CRecorder::postTrackWritePoints@335]  done and not last  i 9 numPoi 11
    2019-07-28 20:27:49.248 INFO  [161752] [CRecorder::compareModes@194]  record_id 990780 event_id 99 track_id 780 photon_id 780 parent_id -1 primary_id -2 reemtrack 0
    2019-07-28 20:27:49.248 INFO  [161752] [CRecorder::compareModes@195] ps:CPhoton slot_constrained 9 seqhis           cbbbbbbbcd seqmat           4111111114 is_flag_done N is_done Y
    2019-07-28 20:27:49.248 INFO  [161752] [CRecorder::compareModes@196] pp:CPhoton slot_constrained 9 seqhis           8bbbbbbbcd seqmat           3111111114 is_flag_done Y is_done Y
    2019-07-28 20:27:49.352 INFO  [161752] [CG4::postpropagate@369] [ (0) ctx CG4Ctx::desc_stats dump_count 0 event_total 100 event_track_count 10000


* note that all these mismatched are 9/11 

Rearranging the CRec::addPoi limit checking avoids this difference.::

    361     bool limited = false ;
    362 
    363     if(!preSkip)
    364     {
    365         limited = addPoi_(new CPoi(pre, preFlag, u_preMat, m_prior_boundary_status, m_ctx._stage, m_origin));
    366     }
    367 
    368     if(lastPost && !limited)
    369     {
    370         limited = addPoi_(new CPoi(post, postFlag, u_postMat, m_boundary_status, m_ctx._stage, m_origin));
    371     }
    ...
    380     bool done = lastPost || limited ;
    381     return done  ;
    382 }
    383 
    384 
    385 bool CRec::addPoi_(CPoi* poi)
    386 {
    387     bool limited = m_poi.size() >= m_ctx.point_limit() ;
    388     if( !limited )
    389     {
    390         m_poi.push_back(poi);
    391     }
    392     return limited  ;
    393 }



But still G4 consuming more in the tail::

    [2019-07-28 21:44:20,252] p164842 {<module>            :tboolean.py:38} CRITICAL -  RC 0x05 0b101 
    [2019-07-28 21:44:20,275] p164842 {check_utaildebug    :ab.py     :204} INFO     - utail mismatch but seqhis matched u.shape:(1000000, 16, 16) w.shape: (131,) 
     i     0 p    1872 ua     0.8095 ub     0.1871  wa  36 wb  45 wd   9 :    1872   1872 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i     1 p   11341 ua     0.2997 ub     0.7499  wa  36 wb  41 wd   5 :   11341  11341 :   :                      TO BT BR BR BR BR BR BR BR BT                      TO BT BR BR BR BR BR BR BR BT   
     i     2 p   14747 ua     0.6346 ub     0.3680  wa  40 wb  46 wd   6 :   14747  14747 :   :                      TO BT SC BR BR BR BR BR BR BR                      TO BT SC BR BR BR BR BR BR BR   
     i     3 p   28413 ua     0.7844 ub     0.0212  wa  36 wb  45 wd   9 :   28413  28413 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i     4 p   29118 ua     0.7897 ub     0.2034  wa  36 wb  45 wd   9 :   29118  29118 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i     5 p   55856 ua     0.8077 ub     0.1347  wa  36 wb  41 wd   5 :   55856  55856 :   :                      TO BT BR BR BR BR BR BR BR BT                      TO BT BR BR BR BR BR BR BR BT   
     i     6 p   65189 ua     0.3840 ub     0.7007  wa  36 wb  45 wd   9 :   65189  65189 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i     7 p   65895 ua     0.8852 ub     0.4157  wa  36 wb  45 wd   9 :   65895  65895 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i     8 p   69653 ua     0.4836 ub     0.3343  wa  40 wb  49 wd   9 :   69653  69653 :   :                      TO BT SC BR BR BR BR BR BR BR                      TO BT SC BR BR BR BR BR BR BR   
     i     9 p   70510 ua     0.8297 ub     0.4156  wa  13 wb  12 wd  -1 :   70510  70510 :   :                                        TO BT BT SA                                        TO BT BT SA   
     i    10 p   70511 ua     0.1619 ub     0.0000  wa  36 wb  -1 wd   0 :   70511  70511 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    11 p   71280 ua     0.5162 ub     0.1054  wa  36 wb  45 wd   9 :   71280  71280 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    12 p   73533 ua     0.7189 ub     0.3708  wa  50 wb  56 wd   6 :   73533  73533 :   :                      TO BT BT SC BT BR BR BR BR BR                      TO BT BT SC BT BR BR BR BR BR   
     i    13 p   76056 ua     0.4942 ub     0.8756  wa  36 wb  45 wd   9 :   76056  76056 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    14 p   81607 ua     0.0474 ub     0.5089  wa  36 wb  45 wd   9 :   81607  81607 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    15 p   86702 ua     0.3869 ub     0.8174  wa  36 wb  45 wd   9 :   86702  86702 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    16 p   97118 ua     0.6177 ub     0.1149  wa  36 wb  45 wd   9 :   97118  97118 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   
     i    17 p   98796 ua     0.6535 ub     0.8339  wa  36 wb  41 wd   5 :   98796  98796 :   :                      TO BT BR BR BR BR BR BR BR BT                      TO BT BR BR BR BR BR BR BR BT   
     i    18 p  107799 ua     0.6279 ub     0.1445  wa  40 wb  46 wd   6 :  107799 107799 :   :                      TO BT SC BR BR BR BR BR BR BR                      TO BT SC BR BR BR BR BR BR BR   
     i    19 p  133597 ua     0.3362 ub     0.4040  wa  36 wb  45 wd   9 :  133597 133597 :   :                      TO BT BR BR BR BR BR BR BR BR                      TO BT BR BR BR BR BR BR BR BR   




::

    495     int bounce = 0 ;
    ...
    525     PerRayData_propagate prd ;
    526 
    527     while( bounce < bounce_max )
    528     {
    ...
    533         bounce++;   // increment at head, not tail, as CONTINUE skips the tail








Trying to reduce the point_limit to bounce_max. Again getting seqhis zeros::


    ab.mal
    aligned   999611/1000000 : 0.9996 : 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 
    maligned     389/1000000 : 0.0004 : 1872,2180,2908,4860,5477,11341,12338,14747,17891,18117,28413,28709,29118,32764,37671,38187,43675,45874,46032,47325,49872,55856,56239,57334,60178 
    slice(0, 25, None)
          0   1872 : * :                      TO BT BR BR BR BR BR BR BR BR                                                ?0? 
          1   2180 : * :                      TO BT BR BR BR BR BR BR BT SA                                                ?0? 
          2   2908 : * :                                  TO BT BR BR BT SA                               TO BT BR BR BR BT SA 
          3   4860 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
          4   5477 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
          5  11341 : * :                      TO BT BR BR BR BR BR BR BR BT                                                ?0? 
          6  12338 : * :                                     TO BT BR BT SA                                                ?0? 
          7  14747 : * :                      TO BT SC BR BR BR BR BR BR BR                                                ?0? 
          8  17891 : * :                                     TO BT BT BT SA                                     TO BT BT BR SA 
          9  18117 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
         10  28413 : * :                      TO BT BR BR BR BR BR BR BR BR                                                ?0? 
         11  28709 : * :                                     TO BT BT BT SA                                     TO BT BT BR SA 
         12  29118 : * :                      TO BT BR BR BR BR BR BR BR BR                                                ?0? 
         13  32764 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
         14  37671 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
         15  38187 : * :                      TO BT BT SC BT BR BR BR BT SA                                                ?0? 
         16  43675 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
         17  45874 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
         18  46032 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
         19  47325 : * :                      TO BT BR BR BR BR BR BR BT SA                                                ?0? 
         20  49872 : * :                      TO BT SC BR BR BR BR BR BT SA                                                ?0? 
         21  55856 : * :                      TO BT BR BR BR BR BR BR BR BT                                                ?0? 
         22  56239 : * :                      TO BT BR BR BR BR BR BR BT SA                                                ?0? 
         23  57334 : * :                      TO BT BR BR BR BR BR BR BT SA                                                ?0? 
         24  60178 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
    .
    [2019-07-28 22:13:54,975] p166900 {code                :seq.py    :171} WARNING  - SeqType.code check [?0?] bad 1 
    [2019-07-28 22:13:54,981] p166900 {code                :seq.py    :171} WARNING  - SeqType.code check [?0?] bad 1 
    ab




    [2019-07-28 22:13:54,987] p166900 {<module>            :tboolean.py:38} CRITICAL -  RC 0x05 0b101 
    [2019-07-28 22:13:55,009] p166900 {check_utaildebug    :ab.py     :204} INFO     - utail mismatch but seqhis matched u.shape:(1000000, 16, 16) w.shape: (7,) 
     i     0 p   73532 ua     0.3248 ub     0.9248  wa  13 wb  12 wd  -1 :   73532  73532 :   :                                        TO BT BT SA                                        TO BT BT SA   
     i     1 p  193420 ua     0.0879 ub     0.9583  wa  17 wb  16 wd  -1 :  193420 193420 :   :                                     TO BT BR BT SA                                     TO BT BR BT SA   
     i     2 p  258609 ua     0.2699 ub     0.5301  wa  35 wb  30 wd  -5 :  258609 258609 :   :                                  TO BT BR BT SC SA                                  TO BT BR BT SC SA   
     i     3 p  583773 ua     0.2128 ub     0.9355  wa  13 wb  12 wd  -1 :  583773 583773 :   :                                        TO BT BT SA                                        TO BT BT SA   
     i     4 p  635008 ua     0.5399 ub     0.5543  wa  13 wb  12 wd  -1 :  635008 635008 :   :                                        TO BT BT SA                                        TO BT BT SA   
     i     5 p  663781 ua     0.6080 ub     0.2725  wa  13 wb  12 wd  -1 :  663781 663781 :   :                                        TO BT BT SA                                        TO BT BT SA   
     i     6 p  892900 ua     0.1056 ub     0.3332  wa  35 wb  25 wd -10 :  892900 892900 :   :                                  TO BT BR BT SC SA                                  TO BT BR BT SC SA   

    In [1]: 






Suspect the zeros come from hard truncate::

    144 bool CWriter::writeStepPoint(const G4StepPoint* point, unsigned flag, unsigned material )
    145 {
    146     m_photon.add(flag, material);  // sets seqhis/seqmat nibbles in current constrained slot  
    147 
    148     bool hard_truncate = m_photon.is_hard_truncate();    
    149 
    150     hard_truncate = false ;  // TEMPORARY KLUDGE 
    151 
    152     



Remember the recording is just for debug 




After adding the last can get bounce_max point limited to give non-zero seqhis:: 

    144 
    145 * *last* argument is only used in --recpoi mode where it prevents 
    146    truncated photons from never being "done" and giving seqhis zeros
    147 
    148 
    149 **/
    150 
    151 bool CWriter::writeStepPoint(const G4StepPoint* point, unsigned flag, unsigned material, bool last )
    152 {
    153     m_photon.add(flag, material);  // sets seqhis/seqmat nibbles in current constrained slot  
    154     
    155 
    156     bool hard_truncate = m_photon.is_hard_truncate();
    157     
    158     bool done = false ;
    159 
    160     if(hard_truncate)
    161     {
    162         done = true ;
    163     }   
    164     else
    165     {
    166         if(m_enabled) writeStepPoint_(point, m_photon );
    167         
    168         m_photon.increment_slot() ;
    169 
    170         done = m_photon.is_done() ;  // caution truncation/is_done may change after increment
    171         
    172         if( (done || last) && m_enabled )
    173         {
    174             writePhoton(point);
    175             if(m_dynamic) m_records_buffer->add(m_dynamic_records);
    176         }   
    177     }       
    178     
    179     
    180     if( flag == BULK_ABSORB )
    181     {
    182         assert( done == true );
    183     }   
    184     
    185     return done ;
    186 }
    187 




::

    ta 19 --tag 2 --msli :1M
    ...

    .
    ab.mal
    aligned   999611/1000000 : 0.9996 : 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24 
    maligned     389/1000000 : 0.0004 : 1872,2180,2908,4860,5477,11341,12338,14747,17891,18117,28413,28709,29118,32764,37671,38187,43675,45874,46032,47325,49872,55856,56239,57334,60178 
    slice(0, 25, None)
          0   1872 : * :                      TO BT BR BR BR BR BR BR BR BR                         TO BT BR BR BR BR BR BR BR 
          1   2180 : * :                      TO BT BR BR BR BR BR BR BT SA                         TO BT BR BR BR BR BR BR BT 
          2   2908 : * :                                  TO BT BR BR BT SA                               TO BT BR BR BR BT SA 
          3   4860 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
          4   5477 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
          5  11341 : * :                      TO BT BR BR BR BR BR BR BR BT                         TO BT BR BR BR BR BR BR BR 
          6  12338 : * :                                     TO BT BR BT SA                         TO BT BR BR BR BR BR BR BR 
          7  14747 : * :                      TO BT SC BR BR BR BR BR BR BR                         TO BT SC BR BR BR BR BR BR 


    ...

    [2019-07-28 22:45:37,554] p170726 {<module>            :tboolean.py:38} CRITICAL -  RC 0x05 0b101 
    [2019-07-28 22:45:37,568] p170726 {check_utaildebug    :ab.py     :204} INFO     - utail mismatch but seqhis matched u.shape:(1000000, 16, 16) w.shape: (7,) 
     i     0 p   73532 ua     0.3248 ub     0.9248  wa  13 wb  12 wd  -1 :   73532  73532 :   :                                        TO BT BT SA                                        TO BT BT SA   
     i     1 p  193420 ua     0.0879 ub     0.9583  wa  17 wb  16 wd  -1 :  193420 193420 :   :                                     TO BT BR BT SA                                     TO BT BR BT SA   
     i     2 p  258609 ua     0.2699 ub     0.5301  wa  35 wb  30 wd  -5 :  258609 258609 :   :                                  TO BT BR BT SC SA                                  TO BT BR BT SC SA   
     i     3 p  583773 ua     0.2128 ub     0.9355  wa  13 wb  12 wd  -1 :  583773 583773 :   :                                        TO BT BT SA                                        TO BT BT SA   
     i     4 p  635008 ua     0.5399 ub     0.5543  wa  13 wb  12 wd  -1 :  635008 635008 :   :                                        TO BT BT SA                                        TO BT BT SA   
     i     5 p  663781 ua     0.6080 ub     0.2725  wa  13 wb  12 wd  -1 :  663781 663781 :   :                                        TO BT BT SA                                        TO BT BT SA   
     i     6 p  892900 ua     0.1056 ub     0.3332  wa  35 wb  25 wd -10 :  892900 892900 :   :                                  TO BT BR BT SC SA                                  TO BT BR BT SC SA   


So these 7 (utail mismatch but seqhis matched) would cause invalid deviations if were using seqhis alone to judge history alignment.     

* probably the explanation for the -1 are zeroSteps and kludges ... need a way to check that 

::

    In [4]: np.where( a.utail == b.utail )[0].shape
    Out[4]: (999707,)

    In [5]: np.where( a.utail != b.utail )[0].shape
    Out[5]: (293,)

    In [6]: np.where( a.seqhis == b.seqhis )[0].shape
    Out[6]: (999611,)

    In [7]: np.where( a.seqhis != b.seqhis )[0].shape
    Out[7]: (389,)


::

    def _get_misutailed(self):
        return np.where(self.a.utail != self.b.utail)[0]


    In [14]: ab.dumpline(ab.misutailed)
          0   1872 : * :                      TO BT BR BR BR BR BR BR BR BR                         TO BT BR BR BR BR BR BR BR 
          1   2908 : * :                                  TO BT BR BR BT SA                               TO BT BR BR BR BT SA 
          2   4860 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
          3   5477 : * :                                     TO BT BT BT SA                                  TO BT BT BT BT SA 
          4  11341 : * :                      TO BT BR BR BR BR BR BR BR BT                         TO BT BR BR BR BR BR BR BR 
          5  12338 : * :                                     TO BT BR BT SA                         TO BT BR BR BR BR BR BR BR 

