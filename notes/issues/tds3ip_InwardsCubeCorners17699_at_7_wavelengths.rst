tds3ip_InwardsCubeCorners17699_at_7_wavelengths
==================================================

Related
-------

* prior :doc:`ok_less_reemission`
* next :doc:`raw_scintillator_material_props`

tds3ip
----------

::

    tds3ip(){
       #local name="RandomSpherical10" 
       #local name="CubeCorners" 
       #local name="CubeCorners10x10" 
       #local name="CubeCorners100x100" 
       local name="InwardsCubeCorners17699"

       local path="$HOME/.opticks/InputPhotons/${name}.npy"
       #local path=/tmp/check_innerwater_bulk_absorb.npy 

       export OPTICKS_EVENT_PFX=tds3ip
       export INPUT_PHOTON_PATH=$path
       export INPUT_PHOTON_REPEAT=10000  
       : 100k repeat falls foul of Geant4 big primary slowdown  
       export INPUT_PHOTON_WAVELENGTH=360,380,400,420,440,460,480
       export EVTMAX=7
     
       #tds3 --dbgseqhis 0x7ccccd   # "TO BT BT BT BT SD"
       #tds3 --dindex 0,1,2,3,4,5

       tds3 

    }


findbndtex
------------

::

    epsilon:opticks blyth$ opticks-f finebnd
    ./integration/tests/tconcentric.bash:    tconcentric-t --bouncemax 15 --recordmax 16 --groupvel --finebndtex $* 
    ./ggeo/GBndLib.cc:        GDomain<double>* finedom = ok->hasOpt("finebndtex") 
    ./ggeo/GBndLib.cc:            LOG(warning) << "--finebndtex option triggers interpolation of material and surface props "  ;
    ./ggeo/GPropertyLib.cc:   // this is normal domain, only with --finebndtex does the finedomain interpolation get load within GBndLib::load
    ./ggeo/GBndLib.hh:When using --finebndtex option GBndLib::load with constituents true
    ./optickscore/OpticksCfg.cc:       ("finebndtex",  "Use 1nm pitch wavelength domain for boundary buffer (ie material and surface properties) obtained by interpolation postcache, see GGeo::loadFromCache");
    epsilon:opticks blyth$ 

::

     073 /**
      74 GBndLib::load
      75 ---------------
      76 
      77 Hmm, finebndtex appears to be done here postcache ?
      78 It surely makes more sense to define a finer domain to use precache.
      79 
      80 **/
      81 
      82 GBndLib* GBndLib::load(Opticks* ok, bool constituents)
      83 {
      84     LOG(LEVEL) << "[" ;
      85     GBndLib* blib = new GBndLib(ok);
      86     
      87     LOG(verbose) ;
      88     
      89     blib->loadIndexBuffer();
      90     
      91     LOG(verbose) << "indexBuffer loaded" ;
      92     blib->importIndexBuffer();
      93     
      94     
      95     if(constituents)
      96     {
      97         GMaterialLib* mlib = GMaterialLib::load(ok);
      98         GSurfaceLib* slib = GSurfaceLib::load(ok);
      99         GDomain<double>* finedom = ok->hasOpt("finebndtex")
     100                             ?
     101                                 mlib->getStandardDomain()->makeInterpolationDomain(Opticks::FINE_DOMAIN_STEP)
     102                             :   
     103                                 NULL
     104                             ;   
     105                             
     106         //assert(0); 
     107         
     108         if(finedom)
     109         {
     110             LOG(warning) << "--finebndtex option triggers interpolation of material and surface props "  ;
     111             GMaterialLib* mlib2 = new GMaterialLib(mlib, finedom );    
     112             GSurfaceLib* slib2 = new GSurfaceLib(slib, finedom );  
     113             
     114             mlib2->setBuffer(mlib2->createBuffer());
     115             slib2->setBuffer(slib2->createBuffer());
     116             
     117             blib->setStandardDomain(finedom);
     118             blib->setMaterialLib(mlib2);





::

     318    m_desc.add_options()
     319        ("finebndtex",  "Use 1nm pitch wavelength domain for boundary buffer (ie material and surface properties) obtained by interpolation postcache, see GGeo::loadFromCache");
     320 






compare first slot splits at 7 wavelengths
---------------------------------------------------

* more deviation in the middle where abslen changing rapidly
* reemission wavlength peaks in that region so thats consistent with more discrepancy after reemission

::

    epsilon:ana blyth$ tds3ip.sh 1      ## with abs.py for loading/presenting multiple events 
    PFX=tds3ip ab.sh 1
     input_photon start wavelength 360 
     cod la       a          af          b          bf     a-b   (a-b)^2/(a+b)  slot 1  (seqhis_splits)  a.itag 1 b.itag -1 
     0x4 AB   16038       0.200       16012       0.200      26           0.021 
     0x5 RE   63926       0.799       63966       0.800     -40           0.013 
     0x6 SC      36       0.000          22       0.000      14           3.379 
     input_photon start wavelength 380 
     cod la       a          af          b          bf     a-b   (a-b)^2/(a+b)  slot 1  (seqhis_splits)  a.itag 2 b.itag -2 
     0x4 AB   16020       0.200       16055       0.201     -35           0.038 
     0x5 RE   63918       0.799       63885       0.799      33           0.009 
     0x6 SC      62       0.001          60       0.001       2           0.033 
     input_photon start wavelength 400 
     cod la       a          af          b          bf     a-b   (a-b)^2/(a+b)  slot 1  (seqhis_splits)  a.itag 3 b.itag -3 
     0x4 AB   15839       0.198       15917       0.199     -78           0.192 
     0x5 RE   63232       0.790       63167       0.790      65           0.033 
     0x6 SC     929       0.012         916       0.011      13           0.092 
     input_photon start wavelength 420 
     cod la       a          af          b          bf     a-b   (a-b)^2/(a+b)  slot 1  (seqhis_splits)  a.itag 4 b.itag -4 
     0x4 AB   13193       0.165       13268       0.166     -75           0.213 
     0x5 RE   12945       0.162       12948       0.162      -3           0.000 
     0x6 SC   46562       0.582       46507       0.581      55           0.033 
     0xc BT    7300       0.091        7277       0.091      23           0.036 
     input_photon start wavelength 440 
     cod la       a          af          b          bf     a-b   (a-b)^2/(a+b)  slot 1  (seqhis_splits)  a.itag 5 b.itag -5 
     0x4 AB   12832       0.160       13027       0.163    -195           1.470 
     0x5 RE    3628       0.045        3709       0.046     -81           0.894 
     0x6 SC   48022       0.600       47669       0.596     353           1.302 
     0xc BT   15518       0.194       15595       0.195     -77           0.191 
     input_photon start wavelength 460 
     cod la       a          af          b          bf     a-b   (a-b)^2/(a+b)  slot 1  (seqhis_splits)  a.itag 6 b.itag -6 
     0x4 AB   15474       0.193       15417       0.193      57           0.105 
     0x5 RE    3147       0.039        3278       0.041    -131           2.671 
     0x6 SC   43913       0.549       43348       0.542     565           3.658 
     0xc BT   17466       0.218       17957       0.224    -491           6.806 
     input_photon start wavelength 480 
     cod la       a          af          b          bf     a-b   (a-b)^2/(a+b)  slot 1  (seqhis_splits)  a.itag 7 b.itag -7 
     0x4 AB   14653       0.183       14695       0.184     -42           0.060 
     0x5 RE    2314       0.029        2264       0.028      50           0.546 
     0x6 SC   41920       0.524       41777       0.522     143           0.244 
     0xc BT   21113       0.264       21264       0.266    -151           0.538 

    In [1]:                                                           




Very rapid change in abslen could explain differences arising from too coarse domain binning
---------------------------------------------------------------------------------------------------


::

    In [20]: run ls.py                                                                                                                                                                                     
    [{__init__            :proplib.py:150} INFO     - names : None 
    [{__init__            :proplib.py:160} INFO     - npath : /usr/local/opticks/geocache/OKX4Test_lWorld0x32a96e0_PV_g4live/g4ok_gltf/a3cbac8189a032341f76682cdb4f47b6/1/GItemList/GMaterialLib.txt 
    [{__init__            :proplib.py:167} INFO     - names : ['LS', 'Steel', 'Tyvek', 'Air', 'Scintillator', 'TiO2Coating', 'Adhesive', 'Aluminium', 'Rock', 'LatticedShellSteel', 'Acrylic', 'PE_PA', 'Vacuum', 'Pyrex', 'Water', 'vetoWater', 'Galactic'] 
        wavelen      rindex      abslen     scatlen    reemprob    groupvel 
         60.000       1.454       0.003     546.429       0.400     206.241 
         80.000       1.454       0.003     546.429       0.400     206.241 
        100.000       1.454       0.003     546.429       0.400     206.241 
        120.000       1.454       0.003     546.429       0.400     192.299 
        140.000       1.664       0.003     546.429       0.400     173.446 
        160.000       1.793       0.003     546.429       0.400     118.988 
        180.000       1.527       0.003     546.429       0.410     139.949 
        200.000       1.618       0.003     547.535       0.420     177.249 
        220.000       1.600       0.198    1415.292       0.477     166.321 
        240.000       1.582       0.392    2283.049       0.538     166.320 
        260.000       1.563       0.586    3150.806       0.599     166.319 
        280.000       1.545       0.781    4018.563       0.660     166.319 
        300.000       1.526       0.975    4887.551       0.721     177.207 
        320.000       1.521       1.169    7505.381       0.782     186.734 
        340.000       1.516       1.364   10123.211       0.800     186.733 
        360.000       1.511       5.664   12741.041       0.800     186.733 
        380.000       1.505      12.239   15358.871       0.801     186.733     
        400.000       1.500     195.518   17976.701       0.800     189.766   ##  absorption very sensitive to wavelength in this range   
        420.000       1.497   40892.633   23161.414       0.497     193.682     
        440.000       1.495   84240.547   29164.996       0.222     195.357     
        460.000       1.494   78284.352   33453.633       0.169     195.915 
        480.000       1.492   92540.648   37742.270       0.135     195.684 
        500.000       1.490  114196.219   43987.516       0.123     195.369 
        520.000       1.488   88688.727   52136.293       0.106     195.275 
        540.000       1.487   91878.211   60285.070       0.089     196.430 
        560.000       1.485   93913.664   75733.656       0.072     198.024 
        580.000       1.485   67581.016   98222.445       0.057     198.572 
        600.000       1.484   46056.891  116999.734       0.048     198.683 
        620.000       1.483   44640.812  132183.031       0.040     198.732 
        640.000       1.482   15488.402  147366.312       0.031     198.733 
        660.000       1.481   20362.018  162549.594       0.023     198.733 
        680.000       1.480   20500.150  177732.875       0.014     199.247 
        700.000       1.480   13182.578  192957.234       0.005     200.349 
        720.000       1.479    7429.221  218677.828       0.000     200.931 
        740.000       1.479    5515.074  244398.406       0.000     200.931 
        760.000       1.479    2898.857  270119.000       0.000     200.931 
        780.000       1.478   10900.813  295839.562       0.000     200.936 
        800.000       1.478    9584.489  321429.000       0.000     201.905 
        820.000       1.478    5822.304  321429.000       0.000     202.823 

    In [21]:                                                                                        




Estimate proportions of AB/SC/RE/BT at different wavelengths in G4 and OK 
------------------------------------------------------------------------------


::

    tds3ip.sh 1 


    In [15]: ab.his                                                                                                                                                                                        
    Out[15]: 
    ab.his
    .       seqhis_ana  cfo:sum  1:g4live:tds3ip   -1:g4live:tds3ip        c2        ab        ba 
    .                              80000     80000      2051.47/239 =  8.58  (pval:0.000 prob:1.000)  
       n             iseq         a         b    a-b       (a-b)^2/(a+b)         a/b                   b/a           [ns] label
    0000               4d     16038     16012     26              0.02         1.002 +- 0.008        0.998 +- 0.008  [2 ] TO AB
    0001           7ccc5d     10411     10358     53              0.14         1.005 +- 0.010        0.995 +- 0.010  [6 ] TO RE BT BT BT SD

    0002              45d      4621      5026   -405             17.00         0.919 +- 0.014        1.088 +- 0.015  [3 ] TO RE AB      ## OK: 10% LESS IMMEDIATE AB after RE 

    0003             4c5d      4817      4026    791             70.75         1.196 +- 0.017        0.836 +- 0.013  [4 ] TO RE BT AB
    0004       bccbccbc5d      3941      4133   -192              4.57         0.954 +- 0.015        1.049 +- 0.016  [10] TO RE BT BR BT BT BR BT BT BR
    0005           8ccc5d      3818      3960   -142              2.59         0.964 +- 0.016        1.037 +- 0.016  [6 ] TO RE BT BT BT SA
    0006            4bc5d      1988      1853    135              4.74         1.073 +- 0.024        0.932 +- 0.022  [5 ] TO RE BT BR AB
    0007          7ccc65d      1992      1839    153              6.11         1.083 +- 0.024        0.923 +- 0.022  [7 ] TO RE SC BT BT BT SD
    0008            8cc5d      1749      1707     42              0.51         1.025 +- 0.024        0.976 +- 0.024  [5 ] TO RE BT BT SA
    0009          7ccc55d      1515      1722   -207             13.24         0.880 +- 0.023        1.137 +- 0.027  [7 ] TO RE RE BT BT BT SD
    0010            4cc5d      1395      1354     41              0.61         1.030 +- 0.028        0.971 +- 0.026  [5 ] TO RE BT BT AB
    0011           4cbc5d      1050      1132    -82              3.08         0.928 +- 0.029        1.078 +- 0.032  [6 ] TO RE BT BR BT AB
    0012             455d       800      1050   -250             33.78         0.762 +- 0.027        1.312 +- 0.041  [4 ] TO RE RE AB
    0013       c6cbccbc5d       896       940    -44              1.05         0.953 +- 0.032        1.049 +- 0.034  [10] TO RE BT BR BT BT BR BT SC BT
    0014          8ccc65d       757       757      0              0.00         1.000 +- 0.036        1.000 +- 0.036  [7 ] TO RE SC BT BT BT SA
    0015             465d       776       716     60              2.41         1.084 +- 0.039        0.923 +- 0.034  [4 ] TO RE SC AB
    0016          4ccbc5d       768       656    112              8.81         1.171 +- 0.042        0.854 +- 0.033  [7 ] TO RE BT BR BT BT AB
    0017          8ccc55d       563       675   -112             10.13         0.834 +- 0.035        1.199 +- 0.046  [7 ] TO RE RE BT BT BT SA
    0018         4bccbc5d       630       546     84              6.00         1.154 +- 0.046        0.867 +- 0.037  [8 ] TO RE BT BR BT BT BR AB
    .                              80000     80000      2051.47/239 =  8.58  (pval:0.000 prob:1.000)  


After fixing reemission wavelength distrib, but still with coarse domain binning::

    In [1]: ab.his                                                                                                                                                                                    
    Out[1]: 
    ab.his
    .       seqhis_ana  cfo:sum  1:g4live:tds3ip   -1:g4live:tds3ip        c2        ab        ba 
    .                              80000     80000      2272.58/238 =  9.55  (pval:0.000 prob:1.000)  
       n             iseq         a         b    a-b       (a-b)^2/(a+b)         a/b                   b/a           [ns] label
    0000               4d     16038     16012     26              0.02         1.002 +- 0.008        0.998 +- 0.008  [2 ] TO AB
    0001           7ccc5d     10797     10358    439              9.11         1.042 +- 0.010        0.959 +- 0.009  [6 ] TO RE BT BT BT SD    ## MAKES WORSE !
    0002              45d      4547      5026   -479             23.97         0.905 +- 0.013        1.105 +- 0.016  [3 ] TO RE AB             ## MAKES WORSE :

    Problems with wavelength distrib and coarse binning must have beeen counteracting each other ?

    0003             4c5d      4360      4026    334             13.30         1.083 +- 0.016        0.923 +- 0.015  [4 ] TO RE BT AB
    0004       bccbccbc5d      4123      4133    -10              0.01         0.998 +- 0.016        1.002 +- 0.016  [10] TO RE BT BR BT BT BR BT BT BR
    0005           8ccc5d      3914      3960    -46              0.27         0.988 +- 0.016        1.012 +- 0.016  [6 ] TO RE BT BT BT SA
    0006            4bc5d      2042      1853    189              9.17         1.102 +- 0.024        0.907 +- 0.021  [5 ] TO RE BT BR AB
    0007          7ccc65d      2047      1839    208             11.13         1.113 +- 0.025        0.898 +- 0.021  [7 ] TO RE SC BT BT BT SD
    0008            8cc5d      1820      1707    113              3.62         1.066 +- 0.025        0.938 +- 0.023  [5 ] TO RE BT BT SA
    0009          7ccc55d      1394      1722   -328             34.53         0.810 +- 0.022        1.235 +- 0.030  [7 ] TO RE RE BT BT BT SD
    0010            4cc5d      1417      1354     63              1.43         1.047 +- 0.028        0.956 +- 0.026  [5 ] TO RE BT BT AB
    0011           4cbc5d      1114      1132    -18              0.14         0.984 +- 0.029        1.016 +- 0.030  [6 ] TO RE BT BR BT AB
    0012       c6cbccbc5d       940       940      0              0.00         1.000 +- 0.033        1.000 +- 0.033  [10] TO RE BT BR BT BT BR BT SC BT
    0013             455d       720      1050   -330             61.53         0.686 +- 0.026        1.458 +- 0.045  [4 ] TO RE RE AB

    0014          8ccc65d       795       757     38              0.93         1.050 +- 0.037        0.952 +- 0.035  [7 ] TO RE SC BT BT BT SA
    0015             465d       794       716     78              4.03         1.109 +- 0.039        0.902 +- 0.034  [4 ] TO RE SC AB
    0016          4ccbc5d       778       656    122             10.38         1.186 +- 0.043        0.843 +- 0.033  [7 ] TO RE BT BR BT BT AB
    0017         4bccbc5d       643       546     97              7.91         1.178 +- 0.046        0.849 +- 0.036  [8 ] TO RE BT BR BT BT BR AB
    0018          8ccc55d       513       675   -162             22.09         0.760 +- 0.034        1.316 +- 0.051  [7 ] TO RE RE BT BT BT SA
    .                              80000     80000      2272.58/238 =  9.55  (pval:0.000 prob:1.000)  








360nm::

    In [1]: a1,b1 = nb_(a.seqhis, 1 ), nb_(b.seqhis, 1 )       ## nibble 1                                                                                                                                                                         
    In [2]: np.unique(a1)                                                                                                                                                                   
    Out[2]: A([4, 5, 6], dtype=uint64)

    In [3]: a.histype.label(np.unique(a1))                                                                                                                                                  
    Out[3]: ['AB', 'RE', 'SC']    ## no sailors 


The first decision in the history starting from 360nm seems in agreement, ie the ammout of initial reemission::

    In [13]: np.unique(a1, return_counts=True)                                                                                                                                                             
    Out[13]: (A([4, 5, 6], dtype=uint64), array([16038, 63926,    36]))

    In [14]: np.unique(b1, return_counts=True)                                                                                                                                                             
    Out[14]: (A([4, 5, 6], dtype=uint64), array([16012, 63966,    22]))


Behaviour after RE goes off-kilter.

* could be the reemission wavelength distrib, OR not fine enough properties as function of wavelength OR both those

* found and fixed binning artifacts in wavelength distrib



Compare wavelength distribution after reemission
--------------------------------------------------

::

    In [1]: ab.sel = "TO RE .."                                                                                                                                                                            

    In [2]: a.his[:20]                                                                                                                                                                                     
    Out[2]: 
    seqhis_ana
    .                     cfo:-  1:g4live:tds3ip 
    .                              63926         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000           7ccc5d        0.163       10411        [6 ] TO RE BT BT BT SD
    0001             4c5d        0.075        4817        [4 ] TO RE BT AB
    0002              45d        0.072        4621        [3 ] TO RE AB
    0003       bccbccbc5d        0.062        3941        [10] TO RE BT BR BT BT BR BT BT BR
    0004           8ccc5d        0.060        3818        [6 ] TO RE BT BT BT SA
    0005          7ccc65d        0.031        1992        [7 ] TO RE SC BT BT BT SD
    0006            4bc5d        0.031        1988        [5 ] TO RE BT BR AB
    0007            8cc5d        0.027        1749        [5 ] TO RE BT BT SA
    0008          7ccc55d        0.024        1515        [7 ] TO RE RE BT BT BT SD
    0009            4cc5d        0.022        1395        [5 ] TO RE BT BT AB
    0010           4cbc5d        0.016        1050        [6 ] TO RE BT BR BT AB
    0011       c6cbccbc5d        0.014         896        [10] TO RE BT BR BT BT BR BT SC BT
    0012             455d        0.013         800        [4 ] TO RE RE AB
    0013             465d        0.012         776        [4 ] TO RE SC AB
    0014          4ccbc5d        0.012         768        [7 ] TO RE BT BR BT BT AB
    0015          8ccc65d        0.012         757        [7 ] TO RE SC BT BT BT SA
    0016         4bccbc5d        0.010         630        [8 ] TO RE BT BR BT BT BR AB
    0017         7ccc665d        0.009         574        [8 ] TO RE SC SC BT BT BT SD
    0018          8ccc55d        0.009         563        [7 ] TO RE RE BT BT BT SA
    .                              63926         1.00 

    In [3]: b.his[:20]                                                                                                                                                                                     
    Out[3]: 
    seqhis_ana
    .                     cfo:-  -1:g4live:tds3ip 
    .                              63966         1.00 
       n             iseq         frac           a    a-b      [ns] label
    0000           7ccc5d        0.162       10358        [6 ] TO RE BT BT BT SD
    0001              45d        0.079        5026        [3 ] TO RE AB
    0002       bccbccbc5d        0.065        4133        [10] TO RE BT BR BT BT BR BT BT BR
    0003             4c5d        0.063        4026        [4 ] TO RE BT AB
    0004           8ccc5d        0.062        3960        [6 ] TO RE BT BT BT SA
    0005            4bc5d        0.029        1853        [5 ] TO RE BT BR AB
    0006          7ccc65d        0.029        1839        [7 ] TO RE SC BT BT BT SD
    0007          7ccc55d        0.027        1722        [7 ] TO RE RE BT BT BT SD
    0008            8cc5d        0.027        1707        [5 ] TO RE BT BT SA
    0009            4cc5d        0.021        1354        [5 ] TO RE BT BT AB
    0010           4cbc5d        0.018        1132        [6 ] TO RE BT BR BT AB
    0011             455d        0.016        1050        [4 ] TO RE RE AB
    0012       c6cbccbc5d        0.015         940        [10] TO RE BT BR BT BT BR BT SC BT
    0013          8ccc65d        0.012         757        [7 ] TO RE SC BT BT BT SA
    0014             465d        0.011         716        [4 ] TO RE SC AB
    0015          8ccc55d        0.011         675        [7 ] TO RE RE BT BT BT SA
    0016          4ccbc5d        0.010         656        [7 ] TO RE BT BR BT BT AB
    0017       ccbccbc55d        0.010         633        [10] TO RE RE BT BR BT BT BR BT BT
    0018       7ccc6cbc5d        0.009         556        [10] TO RE BT BR BT SC BT BT BT SD
    .                              63966         1.00 

    In [4]: a.wl                                                                                                                                                                                           
    Out[4]: A([399.8847, 451.2116, 417.9102, ..., 408.947 , 410.6584, 400.2349], dtype=float32)

    In [5]: a.wl.shape                                                                                                                                                                                     
    Out[5]: (63926,)

    In [6]: b.wl.shape                                                                                                                                                                                     
    Out[6]: (63966,)


    In [11]: a.wl.min(), a.wl.max()                                                                                                                                                                        
    Out[11]: (A(180., dtype=float32), A(800., dtype=float32))

    In [12]: b.wl.min(), b.wl.max()                                                                                                                                                                        
    Out[12]: (A(200.0341, dtype=float32), A(799.7924, dtype=float32))


    In [20]: bins = np.arange(180,820,20)                                                                                                                                                                  
    In [21]: ah = np.histogram(a.wl, bins=bins)                                                                                                                                                            
    In [22]: bh = np.histogram(b.wl, bins=bins)                                                                                                                                                            

    In [31]: for i in range(len(bins)-1): print(" %3.0f:%3.0f  %6d %6d  " % (bins[i],bins[i+1], ah[0][i], bh[0][i] ))                                                                                      
     180:200       2      0  
     200:220      13     83  
     220:240      25     49  
     240:260      20     37  
     260:280      35     30  
     280:300      23     17  
     300:320      20     10  
     320:340      15     16  
     340:360      38     39  
     360:380     221    124  
     380:400    5873   5041  
     400:420   18311  14295  
     420:440   17958  21229  
     440:460   10845  12417  
     460:480    5723   5689  
     480:500    2461   2549  
     500:520    1002   1047  
     520:540     456    446  
     540:560     242    227  
     560:580     133    133  
     580:600      98     99  
     600:620      90     38  
     620:640      82     44  
     640:660      69     43  
     660:680      53     46  
     680:700      36     36  
     700:720      22     41  
     720:740      37     49  
     740:760       7     41  
     760:780      11     21  
     780:800       5     30  




Compare reemission wavelength distrib
----------------------------------------


* qudarap/tests/QCtxTest.py plots the OK one from GPU texture


jsc::

     537          if ( scnt == 0 ){
     538               ScintillationIntegral =
     539                     (G4PhysicsOrderedFreeVector*)((*theFastIntegralTable)(materialIndex));
     540          }
     541          else{
     542               ScintillationIntegral =
     543                     (G4PhysicsOrderedFreeVector*)((*theSlowIntegralTable)(materialIndex));
     544          }
     ...
     593                 // reemission, the sample method need modification
     594                 G4double CIIvalue = G4UniformRand()*
     595                     ScintillationIntegral->GetMaxValue();
     596                 if (CIIvalue == 0.0) {
     597                     // return unchanged particle and no secondaries 
     598                     aParticleChange.SetNumberOfSecondaries(0);
     599                     return G4VRestDiscreteProcess::PostStepDoIt(aTrack, aStep);
     600                    }
     601                 sampledEnergy=
     602                     ScintillationIntegral->GetEnergy(CIIvalue);


Add::

    186 #ifdef WITH_G4OPTICKS
    187        G4double getSampledEnergy(G4int scnt, G4int materialIndex) const ;
    188        G4double getSampledWavelength(G4int scnt, G4int materialIndex) const ;
    189 #endif


Use these from G4OpticksAnaMgr to save 1M wavelength samples direct from DsG4Scintillation process.
Compare to those from texture in qudarap/tests/QCtxTest.py 





 


