CFG4 PMT Edge Double Reflect
==============================

Loading cfg4 tag 2 (0.97,1) ie edge photons with optixviz enabled then 
adjusting near/far and scanparam.z to just look at thin slice.  NB the OptiX
analytic geometry visualized is not the actual (G4) geometry being used :
however the boundary positions look correct.
 
::

    ggv-;ggv-pmt-test --cfg4 --load --optixviz


.. image:: /env/graphics/ggeoview/issues/cfg4-pyrex-br-br.png
   :width: 900px
   :align: center


::

    96.995 100.005
    96.995 100.005
                          2:PmtInBox   -2:PmtInBox           c2 
                    8ccd       391394       391429             0.00  [4 ] TO BT BT SA
                    4ccd        29418        29306             0.21  [4 ] TO BT BT AB
                      4d        22602        22571             0.02  [2 ] TO AB
                     8bd        18671        19186             7.01  [3 ] TO BR SA
                     8cd        10338        10116             2.41  [3 ] TO BT SA
                     4cd         7554         7510             0.13  [3 ] TO BT AB
                   86ccd         4602         4626             0.06  [5 ] TO BT BT SC SA
                     86d         3762         3729             0.15  [3 ] TO SC SA
                     7cd         3233         3181             0.42  [3 ] TO BT SD
                     4bd         1811         1662             6.39  [3 ] TO BR AB

                  8cbbcd            0         3598          3598.00  [6 ] TO BT BR BR BT SA
               8cccccbcd         1885            0          1885.00  [9 ] TO BT BR BT BT BT BT BT SA
                8ccccbcd          762            0           762.00  [8 ] TO BT BR BT BT BT BT SA
              8ccccccbcd          746            0           746.00  [10] TO BT BR BT BT BT BT BT BT SA


                    8c6d          509          440             5.02  [4 ] TO SC BT SA
                8ccc6ccd          386            2           380.04  [8 ] TO BT BT SC BT BT BT SA
                   46ccd          275          310             2.09  [5 ] TO BT BT SC AB
                8cbc6ccd           26          307           237.12  [8 ] TO BT BT SC BT BR BT SA
                  8c6ccd           20          296           241.06  [6 ] TO BT BT SC BT SA



.. image:: /env/graphics/ggeoview/issues/pmt_edge_geometry.png
   :width: 900px
   :align: center


Push out the edge to accentuate the problem, the radial range from 99:100 mm should always do 
this edge clipping.

How are all these things still happening right out at edge ?

::

    98.994 100.005
    98.995 100.005
    WARNING:env.numerics.npy.seq:code bad abbr ?0? 
    WARNING:env.numerics.npy.seq:code sees 1 bad abbr in TO BR SC BT BT BT BT ?0? BT BT 
                          5:PmtInBox   -5:PmtInBox           c2 
                    8ccd       385483       385663             0.04  [4 ] TO BT BT SA
                     8bd        38310        38104             0.56  [3 ] TO BR SA
                    4ccd        29398        29220             0.54  [4 ] TO BT BT AB
                      4d        22949        23089             0.43  [2 ] TO AB
                   86ccd         4633         4570             0.43  [5 ] TO BT BT SC SA
                     4cd         4541         4435             1.25  [3 ] TO BT AB
                  8cbbcd            0         4268          4268.00  [6 ] TO BT BR BR BT SA
                     86d         3802         3702             1.33  [3 ] TO SC SA
                     4bd         3102         3248             3.36  [3 ] TO BR AB
               8cccccbcd         2332            0          2332.00  [9 ] TO BT BR BT BT BT BT BT SA
                8ccccbcd          898            0           898.00  [8 ] TO BT BR BT BT BT BT SA
              8ccccccbcd          887            0           887.00  [10] TO BT BR BT BT BT BT BT BT SA
                    86bd          557          542             0.20  [4 ] TO BR SC SA



Switch to pencil beam in hope for simplification.

::

    if [ "$tag" == "5" ]; then 
        typ=point
        src=99,0,300
        tgt=99,0,0
    fi   


    98.999 98.999
    98.999 98.999
                          5:PmtInBox   -5:PmtInBox           c2 
                    8ccd       406449       404024             7.26  [4 ] TO BT BT SA
                    4ccd        30692        30456             0.91  [4 ] TO BT BT AB
                      4d        22743        23062             2.22  [2 ] TO AB
                     8bd        17024        18367            50.96  [3 ] TO BR SA
                     4cd         6302         6266             0.10  [3 ] TO BT AB
                   86ccd         4984         4866             1.41  [5 ] TO BT BT SC SA
                  8cbbcd            0         4280          4280.00  [6 ] TO BT BR BR BT SA
                     86d         3758         3725             0.15  [3 ] TO SC SA
              8ccccccbcd         3666            0          3666.00  [10] TO BT BR BT BT BT BT BT BT SA
                     4bd         1469         1672            13.12  [3 ] TO BR AB
                    8c6d          498          505             0.05  [4 ] TO SC BT SA
                8ccc6ccd          334            2           328.05  [8 ] TO BT BT SC BT BT BT SA
                   46ccd          321          327             0.06  [5 ] TO BT BT SC AB
                8cbc6ccd           16          301           256.23  [8 ] TO BT BT SC BT BR BT SA
                  8c6ccd           20          295           240.08  [6 ] TO BT BT SC BT SA
                  4cbbcd            0          287           287.00  [6 ] TO BT BR BR BT AB
                    86bd          274          257             0.54  [4 ] TO BR SC SA
                     46d          253          209             4.19  [3 ] TO SC AB
              cccccc6ccd          126          220            25.54  [10] TO BT BT SC BT BT BT BT BT BT




Use circular beam of radous 0.5mm centered in the point (tag 6) for better visibility
Visualize the optix equivalent process "TO BT BR BT BT BT BT BT BT SA" with a single
reflect.

.. image:: /env/graphics/ggeoview/issues/op_pmt_edge_reflect_transmit_closeup.png
   :width: 900px
   :align: center


::

   TO
   BT enter Pyrex
   BR 1st reflect on inner side of lower Pyrex hemi
   BT ... angles look like a reflect but labelled BT 
   BT ??? subsequent transmit out into MO looks like it happens at wrong position

 
Make this explicit with pmt_skimmer.py 


With CPropLib::m_groupvel_kludge=true 1st 4 recs agree, apart from 4th flag BT/BR::

    INFO:env.numerics.npy.evt:Evt seqs ['TO BT BR BR BT SA'] 
    A(Op)
      0 z    300.000    300.000    300.000 r     98.999     98.999     98.999  t      0.098      0.098      0.098    smry m1/m2   4/ 14 MO/Py  -28 ( 27)  13:TO  
      1 z     67.559     67.559     67.559 r     98.999     98.999     98.999  t      1.251      1.251      1.251    smry m1/m2  14/  4 Py/MO   28 ( 27)  12:BT  
      2 z     50.832     50.832     50.832 r    100.372    100.372    100.372  t      1.331      1.331      1.331    smry m1/m2  14/ 11 Py/OV -125 (124)  11:BR  
      3 z     35.551     35.551     35.551 r     93.176     93.176     93.176  t      1.416      1.416      1.416    smry m1/m2  14/  4 Py/MO   28 ( 27)  12:BT  

      4 z      2.005      2.005      2.005 r     81.137     81.137     81.137  t      1.532      1.532      1.532    smry m1/m2   4/ 14 MO/Py  -28 ( 27)  12:BT  
      5 z   -114.115   -114.115   -114.115 r     42.253     42.253     42.253  t      1.953      1.953      1.953    smry m1/m2  14/ 13 Py/Vm  -29 ( 28)  12:BT  
      6 z   -123.875   -123.875   -123.875 r     39.250     39.250     39.250  t      1.990      1.990      1.990    smry m1/m2  14/ 13 Py/Vm  -29 ( 28)  12:BT  
      7 z   -150.810   -150.810   -150.810 r     39.250     39.250     39.250  t      2.051      2.051      2.051    smry m1/m2   4/ 14 MO/Py  -28 ( 27)  12:BT  
      8 z   -169.002   -169.002   -169.002 r     39.250     39.250     39.250  t      2.081      2.081      2.081    smry m1/m2   4/ 12 MO/Rk  124 (123)  12:BT  
      9 z   -300.000   -300.000   -300.000 r     39.250     39.250     39.250  t      2.301      2.301      2.301    smry m1/m2   4/ 12 MO/Rk  124 (123)   8:SA  
    B(G4)
      0 z    300.000    300.000    300.000 r     98.999     98.999     98.999  t      0.098      0.098      0.098    smry m1/m2   4/  0 MO/?0?    0 ( -1)  13:TO  
      1 z     67.559     67.559     67.559 r     98.999     98.999     98.999  t      1.251      1.251      1.251    smry m1/m2  14/  0 Py/?0?    0 ( -1)  12:BT  
      2 z     50.832     50.832     50.832 r    100.372    100.372    100.372  t      1.331      1.331      1.331    smry m1/m2  14/  0 Py/?0?    0 ( -1)  11:BR  
      3 z     35.551     35.551     35.551 r     93.176     93.176     93.176  t      1.416      1.416      1.416    smry m1/m2  14/  0 Py/?0?    0 ( -1)  11:BR  

      4 z     19.181     19.181     19.181 r     89.001     89.001     89.001  t      1.495      1.495      1.495    smry m1/m2   4/  0 MO/?0?    0 ( -1)  12:BT  
      5 z   -300.000   -300.000   -300.000 r     26.569     26.569     26.569  t      3.107      3.107      3.107    smry m1/m2   4/  0 MO/?0?    0 ( -1)   8:SA  



Note that OpaqueVacuum arrives dynamically (from high boundary index), it differs from Vacuum on by 0.1*absorption length::

    simon:npy blyth$ ggv --mat 10
    [2016-Mar-04 17:36:56.130002]:info: GPropertyMap<T>:: 10       material m:OpaqueVacuum k:refractive_index absorption_length scattering_length reemission_prob OpaqueVacuum
                  domain    refractive_index   absorption_length   scattering_length     reemission_prob
                      60                   1               1e+06               1e+06                   0
    simon:npy blyth$ ggv --mat 12
    [2016-Mar-04 17:38:23.829733]:info: GPropertyMap<T>:: 12       material m:Vacuum k:refractive_index absorption_length scattering_length reemission_prob Vacuum
                  domain    refractive_index   absorption_length   scattering_length     reemission_prob
                      60                   1               1e+07               1e+06                   0




FIXED : a TIR bug 
---------------------

* reported in npy-/pmt_skimmer.py 




