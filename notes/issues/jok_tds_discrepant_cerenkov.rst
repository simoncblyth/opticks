jok_tds_discrepant_cerenkov
===============================

workflow
-----------

Workstation::

    ~/j/okjob.sh                      # original junosw+opticks with 3 min init 

Laptop::

    ~/j/jtds/jtds.sh grab
    ~/j/jtds/jtds.sh ana 


Lack CK histories other than crazy "CK"::

    In [11]: ab.a.qu[:1000]
    Out[11]: 
    array([b'CK                                                                                              ',
           b'SI AB                                                                                           ',
           b'SI BT AB                                                                                        ',
           b'SI BT BR BT AB                                                                                  ',
           b'SI BT BR BT BT BT BT BR BT BT BT DR BT SA                                                       ',
           b'SI BT BR BT BT BT BT BT BT BT SD                                                                ',
           b'SI BT BR BT SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT BR BT BT BT BT BT BT RE RE RE BT BT ',
           b'SI BT BR BT SC BT BT BT BT BT BT SA                                                             ',
           b'SI BT BT AB                                                                                     ',

::

    In [19]: w = a.q_startswith("CK") ; w.shape
    Out[19]: (74,)

    In [20]: np.unique(a.q[w])
    Out[20]: array([b'CK                                                                                              '], dtype='|S96')


    In [21]: wb = b.q_startswith("CK")

    In [22]: wb.shape
    Out[22]: (74,)

    In [23]: b.q[wb]
    Out[23]:
    array([[b'CK RE BT BT BT BT BT BT SD                                                                      '],
           [b'CK SC SC SC SC BT BT BT BT BT BT SD                                                             '],
           [b'CK AB                                                                                           '],
           [b'CK RE SC SC SC BT BT BT BT BT BT SA                                                             '],
           [b'CK SC BT BT AB                                                                                  '],
           [b'CK RE RE AB                                                                                     '],
           [b'CK AB                                                                                           '],
           [b'CK RE BT BT BT BT BT BT BT SR BR SA                                                             '],
           [b'CK AB                                                                                           '],

    In [24]: wb    ## 6/7 CK gensteps 
    Out[24]: 
    array([1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1760, 1761, 1762, 1763, 1764, 1765, 1766,
           2500, 2501, 2502, 2503, 2504, 2505, 2506, 2507, 4345, 4346, 4347, 4348, 4349, 4350, 4351, 4352, 4353, 4354, 4355, 4356, 4357, 4358, 4359, 4360, 4361, 4362, 4363, 4364, 4365, 5566, 5567, 5568,
           5569, 5570, 5571, 5572, 5573, 6270, 6271, 6272, 6273, 6274])


::

    034 struct scerenkov
     35 {
     36     // ctrl
     37     unsigned gentype ;   // formerly Id
     38     unsigned trackid ;   // formerly ParentId
     39     unsigned matline ;   // formerly MaterialIndex, used by qbnd::boundary_lookup 
     40     unsigned numphoton ; // formerly NumPhotons 
     41 

    019 enum
     20 {
     21     OpticksGenstep_INVALID                  = 0,
     22     OpticksGenstep_G4Cerenkov_1042          = 1,
     23     OpticksGenstep_G4Scintillation_1042     = 2,
     24     OpticksGenstep_DsG4Cerenkov_r3971       = 3,
     25     OpticksGenstep_DsG4Scintillation_r3971  = 4,
     26     OpticksGenstep_DsG4Scintillation_r4695  = 5,
     27     OpticksGenstep_TORCH                    = 6,
     28     OpticksGenstep_FABRICATED               = 7,
     29     OpticksGenstep_EMITSOURCE               = 8,
     30     OpticksGenstep_NATURAL                  = 9,
     31     OpticksGenstep_MACHINERY                = 10,
     32     OpticksGenstep_G4GUN                    = 11,
     33     OpticksGenstep_PRIMARYSOURCE            = 12,
     34     OpticksGenstep_GENSTEPSOURCE            = 13,
     35     OpticksGenstep_CARRIER                  = 14,
     36     OpticksGenstep_CERENKOV                 = 15,
     37     OpticksGenstep_SCINTILLATION            = 16,
     38     OpticksGenstep_FRAME                    = 17,
     39     OpticksGenstep_G4Cerenkov_modified      = 18,
     40     OpticksGenstep_INPUT_PHOTON             = 19,
     41     OpticksGenstep_NumType                  = 20
     42 };




    In [35]: a.f.genstep.view(np.int32)[:,0]
    Out[35]: 
    array([[    5,     1,    -1,     2],
           [    5,     7,    -1,    20],
           [    5,     7,    -1,     7],
           [    5,     7,    -1,     1],
           [    5,     7,    -1,     1],
           [    5,     5,    -1,   175],
           [    5,     5,    -1,    48],
           [    5,     5,    -1,    15],
           [    5,     5,    -1,     7],
           [    5,     4,    -1,   482],
           [    5,     4,    -1,   134],
           [    5,     4,    -1,    45],
           [    5,     4,    -1,    21],


::

    In [36]: igs = a.f.genstep.view(np.int32)[:,0]

    In [37]: igs[igs[:,0] == 18]
    Out[37]: 
    array([[18,  3, -1, 25],
           [18,  3, -1,  7],
           [18,  3, -1,  8],
           [18,  2, -1, 21],
           [18,  2, -1,  8],
           [18,  2, -1,  5]], dtype=int32)

    In [38]: igs[igs[:,0] == 18][:,3]
    Out[38]: array([25,  7,  8, 21,  8,  5], dtype=int32)

    In [39]: igs[igs[:,0] == 18][:,3].sum()
    Out[39]: 74



HMM : the CK genstep matline are all -1 
-------------------------------------------


Issue with CK dir and pol for all 74::

    In [32]: a.f.record[w,0]
    Out[32]: 
    array([[[  53.762,  -89.348, -212.53 ,    0.815],
            [     inf,     -inf,     -inf,    0.   ],
            [     nan,     -inf,      inf,  248.952],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[  53.717,  -89.182, -212.499,    0.815],
            [     inf,     -inf,     -inf,    0.   ],
            [     inf,      nan,      inf,  754.066],
            [   0.   ,    0.   ,    0.   ,    0.   ]],

           [[  53.676,  -89.034, -212.471,    0.814],
            [     inf,     -inf,     -inf,    0.   ],
            [    -inf,      nan,     -inf,   91.347],
            [   0.   ,    0.   ,    0.   ,    0.   ]],


::

    2023-11-27 11:08:39.559 INFO  [249238] [QEvent::setGenstepUpload@309] ]
    2023-11-27 11:08:39.559 INFO  [249238] [QEvent::setGenstep@198] ]
    //qcerenkov::wavelength_sampled_bndtex idx   6344 sampledRI   0.000 cosTheta     inf sin2Theta   0.000 wavelength 128.340 count 100 
    //qcerenkov::wavelength_sampled_bndtex idx   1771 sampledRI   0.000 cosTheta     inf sin2Theta   0.000 wavelength 105.259 count 100 
    //qcerenkov::wavelength_sampled_bndtex idx   1772 sampledRI   0.000 cosTheta     inf sin2Theta   0.000 wavelength 227.446 count 100 
    //qcerenkov::wavelength_sampled_bndtex idx   1773 sampledRI   0.000 cosTheta     inf sin2Theta   0.000 wavelength  93.682 count 100 
    //qcerenkov::wavelength_sampled_bndtex idx   1774 sampledRI   0.000 cosTheta     inf sin2Theta   0.000 wavelength 256.129 count 100 
    //qcerenkov::wavelength_sampled_bndtex idx   1775 sampledRI   0.000 cosTheta     inf sin2Theta   0.000 wavelength 106.167 count 100 
    //qcerenkov::wavelength_sampled_bndtex idx   1776 sampledRI   0.000 cosTheta     inf sin2Theta   0.000 wavelength  84.038 count 100 
    //qcerenkov::wavelength_sampled_bndtex idx   1777 sampledRI   0.000 cosTheta     inf sin2Theta   0.000 wavelength 284.152 count 100 
    //qcerenkov::wavelength_sampled_bndtex idx   1778 sampledRI   0.000 cosTheta     inf sin2Theta   0.000 wavelength 133.044 count 100 
    //qcerenkov::wavelength_sampled_bndtex idx   5732 sampledRI   0.000 cosTheta     inf sin2Theta   0.000 wavelength  88.715 count 100 
    //qcerenkov::wavelength_sampled_bndtex idx   5733 sampledRI   0.000 cosTheta     inf sin2Theta   0.000 wavelength 125.505 count 100 
    //qcerenkov::wavelength_sampled_bndtex idx   5734 sampledRI   0.000 cosTheta     inf sin2Theta   0.000 wavelength 110.706 count 100 
    //qcerenkov::wavelength_sampled_bndtex idx   5735 sampledRI   0.000 cosTheta     inf sin2Theta   0.000 wavelength  81.378 count 100 
    //qcerenkov::wavelength_sampled_bndtex idx   5736 sampledRI   0.000 cosTheta     inf sin2Theta   0.000 wavelength 165.834 count 100 
    //qcerenkov::wavelength_sampled_bndtex idx   5737 sampledRI   0.000 cosTheta     inf sin2Theta   0.000 wavelength 668.685 count 100 
    //qcerenkov::wavelength_sampled_bndtex idx   5738 sampledRI   0.000 cosTheta     inf sin2Theta   0.000 wavelength  85.890 count 100 
    //qcerenkov::wavelength_sampled_bndtex idx   5739 sampledRI   0.000 cosTheta     inf sin2Theta   0.000 wavelength 132.738 count 100 
    //qcerenkov::wavelength_sampled_bndtex idx   2938 sampledRI   0.000 cosTheta     inf sin2Theta   0.000 wavelength 260.668 count 100 
    //qcerenkov::wavelength_sampled_bndtex idx   1095 sampledRI   0.000 cosTheta     inf sin2Theta   0.000 wavelength 114.176 count 100 


::

    287 inline QCERENKOV_METHOD void qcerenkov::wavelength_sampled_bndtex(float& wavelength, float& cosTheta, float& sin2Theta, curandStateXORWOW& rng, const scerenkov& gs, int     idx, int gsid ) const
    288 {
    289     //printf("//qcerenkov::wavelength_sampled_bndtex bnd %p gs.matline %d \n", bnd, gs.matline ); 
    290     float u0 ;
    291     float u1 ;
    292     float w ;
    293     float sampledRI ;
    294     float u_maxSin2 ;
    295 
    296     unsigned count = 0 ;
    297 
    298     do {
    299         u0 = curand_uniform(&rng) ;
    300 
    301         w = gs.Wmin + u0*(gs.Wmax - gs.Wmin) ;
    302 
    303         wavelength = gs.Wmin*gs.Wmax/w ; // reciprocalization : arranges flat energy distribution, expressed in wavelength 
    304 
    305         float4 props = bnd->boundary_lookup(wavelength, gs.matline, 0u);
    306 
    307         sampledRI = props.x ;
    308 
    309         //printf("//qcerenkov::wavelength_sampled_bndtex count %d wavelength %10.4f sampledRI %10.4f \n", count, wavelength, sampledRI );  
    310 
    311         cosTheta = gs.BetaInverse / sampledRI ;
    312 
    313         sin2Theta = fmaxf( 0.f, (1.f - cosTheta)*(1.f + cosTheta));
    314 
    315         u1 = curand_uniform(&rng) ;
    316 
    317         u_maxSin2 = u1*gs.maxSin2 ;
    318 
    319         count += 1 ;
    320 
    321     } while ( u_maxSin2 > sin2Theta && count < 100 );
    322 
    323     if(count > 50)
    324     printf("//qcerenkov::wavelength_sampled_bndtex idx %6d sampledRI %7.3f cosTheta %7.3f sin2Theta %7.3f wavelength %7.3f count %d \n",
    325               idx , sampledRI, cosTheta, sin2Theta, wavelength, count );
    326 }
    327 






    PIDX=1090 GDB=1 ~/j/okjob.sh 





    In [10]: ab.b.qu[:1000]
    Out[10]: 
    array([b'CK AB                                                                                           ',
           b'CK BT BT BT BT BT BT BR BT BT BT BT BT BR BT AB                                                 ',
           b'CK BT BT BT BT BT BT BT SA                                                                      ',
           b'CK BT BT BT BT BT BT SA                                                                         ',
           b'CK RE AB                                                                                        ',
           b'CK RE BT AB                                                                                     ',
           b'CK RE BT BT BT BT BT BT BT SR BR SA                                                             ',
           b'CK RE BT BT BT BT BT BT BT SR BT BT BT BT BT BT BT BT BT BT BT SA                               ',
           b'CK RE BT BT BT BT BT BT SA                                                                      ',
           b'CK RE BT BT BT BT BT BT SD                                                                      ',
           b'CK RE RE AB                                                                                     ',
           b'CK RE RE RE BT BT BT BT BT BT SA                                                                ',
           b'CK RE RE RE RE AB                                                                               ',
           b'CK RE RE RE RE BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT AB                                  ',
           b'CK RE RE RE RE RE SC SC BT BT BT BT BT BT BT SD                                                 ',
           b'CK RE RE SC SC BT BT BT BT BR BT BT BT BT BT BT SR BR SR BT SA                                  ',
           b'CK RE RE SC SC SC SC SC BT BT BT BT BT BT SA                                                    ',
           b'CK RE SC AB                                                                                     ',
           b'CK RE SC BT BT BT BT BT BT SA                                                                   ',
           b'CK RE SC BT BT BT BT BT SA                                                                      ',
           b'CK RE SC RE AB                                                                                  ',
           b'CK RE SC RE BT BT BT BT BT BT SD                                                                ',
           b'CK RE SC SC BT BT BT BT BT BT SD                                                                ',
           b'CK RE SC SC RE SC SC BT BR BT BT BR BT BT BR BT BT BR BT BT BR BT BT BR BT SC BT BT BT BT SD    ',
           b'CK RE SC SC SC BT BT BT BT BT BT SA                                                             ',
           b'CK SC BT BT AB                                                                                  ',
           b'CK SC BT BT BT BT BT BT SD                                                                      ',
           b'CK SC SC AB                                                                                     ',
           b'CK SC SC BT BT BT BT SA                                                                         ',
           b'CK SC SC SC SC BT BT BT BT BT BT SD                                                             ',
           b'SI AB                                                                                           ',
           b'SI BT AB                                                                                        ',
           b'SI BT BR BT AB                                                                                  ',
           b'SI BT BR BT BT BT BT BT BT BT SD                                                                ',
           b'SI BT BR BT BT BT DR BT DR BT BT BR DR BT BT BT BT BT BT BT BT BT BT BT BT BT BT BT SR BT DR BR ',
           b'SI BT BR BT SC AB                                                                               ',
           b'SI BT BR BT SC SC BT BT BT BT BT BT BR BT BT BT BT AB                                           ',




Try to get clever with input gensteps
----------------------------------------

Workstation::

    ~/opticks/CSGOptiX/cxs_min.sh     # configured to use gensteps from original 

Laptop::

    ~/opticks/CSGOptiX/cxs_min.sh grab 
    ~/opticks/CSGOptiX/cxs_min.sh ana


HUH using the original input gensteps in cxs_min.sh does not have the issue::

    #srm=SRM_DEFAULT
    #srm=SRM_TORCH
    #srm=SRM_INPUT_PHOTON
    srm=SRM_INPUT_GENSTEP
    #srm=SRM_GUN
    export OPTICKS_RUNNING_MODE=$srm

    echo $BASH_SOURCE OPTICKS_RUNNING_MODE $OPTICKS_RUNNING_MODE

    if [ "$OPTICKS_RUNNING_MODE" == "SRM_INPUT_GENSTEP" ]; then 

        igs=$TMP/GEOM/$GEOM/jok-tds/ALL0/p001/genstep.npy 
        export OPTICKS_INPUT_GENSTEP=$igs
        [ ! -f "$igs" ] && echo $BASH_SOURCE : FATAL : NO SUCH PATH : igs $igs && exit 1


Possibly the gensteps get uploaded before some material index to 
matline lookups are done ? 

::

    In [4]: a.f.genstep.view(np.int32)[:,0]
    Out[4]: 
    array([[    5,     1,     0,     2],
           [    5,     7,     0,    20],
           [    5,     7,     0,     7],
           [    5,     7,     0,     1],
           [    5,     7,     0,     1],
           [    5,     5,     0,   175],
           [    5,     5,     0,    48],
           [    5,     5,     0,    15],
           [    5,     5,     0,     7],
           [    5,     4,     0,   482],
           [    5,     4,     0,   134],
           [    5,     4,     0,    45],




The gs.matline is zero in the gensteps that work and -1 in those that dont::

    /home/blyth/opticks/CSGOptiX/cxs_min.sh : run : delete prior LOGFILE CSGOptiXSMTest.log
    2023-11-27 11:38:26.169 INFO  [303504] [CSGOptiX::SimulateMain@175]  OPTICKS_NUM_EVENT=3 OPTICKS_RUNNING_MODE=SRM_INPUT_GENSTEP SEventConfig::IsRunningModeTorch() NO 
    //qcerenkov::wavelength_sampled_bndtex idx   6272 sampledRI   1.000 cosTheta   1.460 sin2Theta   0.000 wavelength  81.404 count 100 matline 0 
    //qcerenkov::wavelength_sampled_bndtex idx   6273 sampledRI   1.000 cosTheta   1.460 sin2Theta   0.000 wavelength  94.631 count 100 matline 0 
    //qcerenkov::wavelength_sampled_bndtex idx   6274 sampledRI   1.000 cosTheta   1.460 sin2Theta   0.000 wavelength  81.146 count 100 matline 0 
    //qcerenkov::wavelength_sampled_bndtex idx   6270 sampledRI   1.000 cosTheta   1.460 sin2Theta   0.000 wavelength 170.250 count 100 matline 0 
    //qcerenkov::wavelength_sampled_bndtex idx   6271 sampledRI   1.000 cosTheta   1.460 sin2Theta   0.000 wavelength  95.640 count 100 matline 0 
    //qcerenkov::wavelength_sampled_bndtex idx   5568 sampledRI   1.000 cosTheta   1.342 sin2Theta   0.000 wavelength 797.862 count 100 matline 0 
    //qcerenkov::wavelength_sampled_bndtex idx   5569 sampledRI   1.000 cosTheta   1.342 sin2Theta   0.000 wavelength 121.048 count 100 matline 0 
    //qcerenkov::wavelength_sampled_bndtex idx   5570 sampledRI   1.000 cosTheta   1.342 sin2Theta   0.000 wavelength  99.718 count 100 matline 0 




See variety of CK histories::

    In [4]: a.qu[:100]
    Out[4]: 
    array([b'CK AB                                                                                           ',
           b'CK BT BT BT BT BT BT BR BT BT BT BT BT BT SC SC AB                                              ',
           b'CK BT BT DR BT DR BT BT SA                                                                      ',
           b'CK BT BT SA                                                                                     ',
           b'CK RE AB                                                                                        ',
           b'CK RE BT AB                                                                                     ',
           b'CK RE BT BT BT BT BT BR BR AB                                                                   ',
           b'CK RE BT BT BT BT BT BT BT SA                                                                   ',
           b'CK RE BT BT BT BT BT BT SA                                                                      ',
           b'CK RE BT BT BT BT BT BT SD                                                                      ',
           b'CK RE RE AB                                                                                     ',
           b'CK RE RE BT BT BT BR BT BT BT BT DR BT BR BR BR BR BR BR BR BR SA                               ',
           b'CK RE RE BT BT BT BT SD                                                                         ',
           b'CK RE RE BT BT SA                                                                               ',
           b'CK RE RE RE BT BT BT BT BT BT BT SD                                                             ',
           b'CK RE RE RE RE RE RE SC RE RE RE SC AB                                                          ',
           b'CK RE RE SC BT BT BT BT BT BT SD                                                                ',
           b'CK RE RE SC BT BT SA                                                                            ',
           b'CK RE RE SC RE BT BT BT BT BT BT SA                                                             ',
           b'CK RE RE SC SC SC AB                                                                            ',
           b'CK RE SC AB                                                                                     ',
           b'CK RE SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SD                                        ',
           b'CK RE SC BT BT BT BT BT BT BT SR SA                                                             ',
           b'CK RE SC BT BT BT BT BT BT SA                                                                   ',
           b'CK RE SC BT BT BT BT BT BT SD                                                                   ',
           b'CK RE SC BT BT BT SA                                                                            ',
           b'CK RE SC SC AB                                                                                  ',
           b'CK RE SC SC BT BT BT BT BT BT SD                                                                ',
           b'CK SC BT BT SA                                                                                  ',
           b'SI AB                                                                                           ',
           b'SI BT AB                                                                                        ',
           b'SI BT BR BT AB                                                                                  ',
           b'SI BT BR BT BT BT BT BR BT BT BT DR BT SA                                                       ',
           b'SI BT BR BT BT BT BT BT BT BT SD                                                                ',
           b'SI BT BR BT SC BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT BR BT BT BT BT BT BT RE RE RE BT BT ',





issue : CK broken => bad chi2
------------------------------

~/j/jtds/jtds.sh ana::

    QCF qcf :  
    a.q 8955 b.q 8955 lim slice(None, None, None) 
    c2sum :   178.9178 c2n :    60.0000 c2per:     2.9820  C2CUT:   30 
    c2sum/c2n:c2per(C2CUT)  178.92/60:2.982 (30) pv[0.00,< 0.05 : NOT:null-hyp ] 

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][0:40]  ## A-B history frequency chi2 comparison 
    [[' 0' 'SI AB                                                                                          ' ' 0' '  1382   1401' ' 0.1297' '     7      1']
     [' 1' 'SI BT BT BT BT BT BT SD                                                                        ' ' 1' '   472    460' ' 0.1545' '    13     13']
     [' 2' 'SI BT BT BT BT BT BT SA                                                                        ' ' 2' '   460    454' ' 0.0394' '    56     10']
     [' 3' 'SI RE AB                                                                                       ' ' 3' '   405    374' ' 1.2336' '    28     12']
     [' 4' 'SI SC AB                                                                                       ' ' 4' '   313    259' ' 5.0979' '    99      0']
     [' 5' 'SI SC BT BT BT BT BT BT SD                                                                     ' ' 5' '   210    221' ' 0.2807' '     8     50']
     [' 6' 'SI SC BT BT BT BT BT BT SA                                                                     ' ' 6' '   189    197' ' 0.1658' '   117     23']
     [' 7' 'SI RE BT BT BT BT BT BT SD                                                                     ' ' 7' '   167    153' ' 0.6125' '     9     11']
     [' 8' 'SI BT BT SA                                                                                    ' ' 8' '   140    155' ' 0.7627' '    97    254']
     [' 9' 'SI RE BT BT BT BT BT BT SA                                                                     ' ' 9' '   130    148' ' 1.1655' '    71    108']
     ['10' 'SI RE RE AB                                                                                    ' '10' '   141    128' ' 0.6283' '    38    110']
     ['11' 'SI SC SC AB                                                                                    ' '11' '    95    102' ' 0.2487' '    67    220']
     ['12' 'SI RE SC AB                                                                                    ' '12' '    99     81' ' 1.8000' '   187    114']
     ['13' 'SI BT BT AB                                                                                    ' '13' '    85     77' ' 0.3951' '    42     36']
     ['14' 'SI SC SC BT BT BT BT BT BT SA                                                                  ' '14' '    72     82' ' 0.6494' '    55    176']
     ['15' 'SI BT BT BT BT BT BT BT SR SA                                                                  ' '15' '    78     76' ' 0.0260' '    49     80']
     ['16' 'CK                                                                                             ' '16' '    74      0' '74.0000' '  1090     -1']
     ['17' 'SI BT BT BT BT BT BT BT SA                                                                     ' '17' '    57     74' ' 2.2061' '    37    194']
     ['18' 'SI SC SC BT BT BT BT BT BT SD                                                                  ' '18' '    69     68' ' 0.0073' '    47    266']
     ['19' 'SI RE SC BT BT BT BT BT BT SD                                                                  ' '19' '    58     61' ' 0.0756' '   289    131']
     ['20' 'SI RE SC BT BT BT BT BT BT SA                                                                  ' '20' '    49     58' ' 0.7570' '    50    142']
     ['21' 'SI BT BT BT BT SD                                                                              ' '21' '    37     56' ' 3.8817' '   295    510']
     ['22' 'SI RE RE BT BT BT BT BT BT SA                                                                  ' '22' '    43     52' ' 0.8526' '   218     47']
     ['23' 'SI RE BT BT SA                                                                                 ' '23' '    48     42' ' 0.4000' '    30     45']
     ['24' 'SI SC BT BT SA                                                                                 ' '24' '    40     45' ' 0.2941' '  1076    390']
     ['25' 'SI RE RE BT BT BT BT BT BT SD                                                                  ' '25' '    45     43' ' 0.0455' '    19     93']
     ['26' 'SI BT BT BT SA                                                                                 ' '26' '    40     44' ' 0.1905' '   190    304']
     ['27' 'SI SC BT BT BT BT BT BT BT SA                                                                  ' '27' '    43     26' ' 4.1884' '    84    173']
     ['28' 'SI BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SD                                             ' '28' '    42     34' ' 0.8421' '   128    177']
     ['29' 'SI SC SC SC AB                                                                                 ' '29' '    42     31' ' 1.6575' '  1059     65']
     ['30' 'SI BT AB                                                                                       ' '30' '    35     42' ' 0.6364' '    26    105']
     ['31' 'SI BT BT BT BT BT BT BR BT BT BT BT BT BT BT BT SA                                             ' '31' '    39     30' ' 1.1739' '    14    620']
     ['32' 'SI BT BT DR BT SA                                                                              ' '32' '    37     37' ' 0.0000' '   338     41']
     ['33' 'SI RE RE RE AB                                                                                 ' '33' '    37     37' ' 0.0000' '   983    419']
     ['34' 'SI SC BT BT BT BT BT BT BT SR SA                                                               ' '34' '    36     25' ' 1.9836' '   757   1383']
     ['35' 'SI SC BT BT AB                                                                                 ' '35' '    34     35' ' 0.0145' '   331     15']
     ['36' 'CK AB                                                                                          ' '36' '     0     35' '35.0000' '    -1   1092']
     ['37' 'SI RE BT BT AB                                                                                 ' '37' '    32     33' ' 0.0154' '   686    116']
     ['38' 'SI RE RE SC AB                                                                                 ' '38' '    32     33' ' 0.0154' '   438    175']
     ['39' 'SI SC RE AB                                                                                    ' '39' '    33     29' ' 0.2581' '   225     97']]

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][bzero]  ## in A but not B 
    [['16' 'CK                                                                                             ' '16' '    74      0' '74.0000' '  1090     -1']]

    np.c_[siq,_quo,siq,sabo2,sc2,sabo1][azero]  ## in B but not A 
    [['36' 'CK AB                                                                                          ' '36' '     0     35' '35.0000' '    -1   1092']]
    ]----- repr(ab) 





matline
---------


::

    epsilon:production blyth$ opticks-f matline


    ./opticksgeo/OpticksGen.cc:just need to avoid trying to translate the matline later.
    ./opticksgeo/OpticksGen.cc:   unsigned int matline = m_blib->getMaterialLine(material);
    ./opticksgeo/OpticksGen.cc:   gs->setMaterialLine(matline);  
    ./opticksgeo/OpticksGen.cc:              << " matline " << matline


    ./sysrap/squad.h:    SQUAD_METHOD unsigned matline() const {   return q0.u.z ; }
    ./sysrap/squad.h:    SQUAD_METHOD void set_matline(  unsigned ml) { q0.u.z = ml ; }

    ./sysrap/SEvt.hh:index and photon offset in addition to  gentype/trackid/matline/numphotons 

    ./sysrap/scarrier.h:   SCARRIER_METHOD static void FillGenstep( scarrier& gs, unsigned matline, unsigned numphoton_per_genstep, bool dump ) ; 
    ./sysrap/scarrier.h:inline void scarrier::FillGenstep( scarrier& gs, unsigned matline, unsigned numphoton_per_genstep, bool dump ) 

    ./sysrap/scerenkov.h:    unsigned matline ;   // formerly MaterialIndex, used by qbnd::boundary_lookup 
    ./sysrap/scerenkov.h:   static void FillGenstep( scerenkov& gs, unsigned matline, unsigned numphoton_per_genstep, bool dump ) ; 
    ./sysrap/scerenkov.h:* NB matline is crucial as that determines which materials RINDEX is used 
    ./sysrap/scerenkov.h:inline void scerenkov::FillGenstep( scerenkov& gs, unsigned matline, unsigned numphoton_per_genstep, bool dump )
    ./sysrap/scerenkov.h:    gs.matline = matline ; 

    ./sysrap/SEvt.cc:    unsigned matline_ = q_.matline(); 
    ./sysrap/SEvt.cc:    if(matline_ >= G4_INDEX_OFFSET )
    ./sysrap/SEvt.cc:        unsigned mtindex = matline_ - G4_INDEX_OFFSET ; 
    ./sysrap/SEvt.cc:        int matline = cf ? cf->lookup_mtline(mtindex) : 0 ;
    ./sysrap/SEvt.cc:        q.set_matline(matline); 
    ./sysrap/SEvt.cc:            << " matline_ " << matline_ 
    ./sysrap/SEvt.cc:            << " matline " << matline


     SEvt::addGenstep sets the matline 

    1929     if(matline_ >= G4_INDEX_OFFSET )
    1930     {
    1931         unsigned mtindex = matline_ - G4_INDEX_OFFSET ;
    1932         int matline = cf ? cf->lookup_mtline(mtindex) : 0 ;
    1933         q.set_matline(matline);
    1934 
    1935         LOG_IF(info, is_cerenkov_gs )
    1936             << " is_cerenkov_gs " << ( is_cerenkov_gs ? "YES" : "NO " )
    1937             << " cf " << ( cf ? "YES" : "NO " )
    1938             << " gentype " << gentype
    1939             << " mtindex " << mtindex
    1940             << " matline_ " << matline_
    1941             << " matline " << matline
    1942             ;
    1943     }


    0785 /**
     786 SEvt::setGeo
     787 -------------
     788 
     789 SGeo is a protocol for geometry access fulfilled by CSGFoundry (and formerly by GGeo)
     790 
     791 Canonical invokation is from G4CXOpticks::setGeometry 
     792 This connection between the SGeo geometry and SEvt is what allows 
     793 the appropriate instance frame to be accessed. That is vital for 
     794 looking up the sensor_identifier and sensor_index.  
     795 
     796 TODO: replace this with stree.h based approach  
     797 
     798 **/
     799 
     800 void SEvt::setGeo(const SGeo* cf_)
     801 {
     802     cf = cf_ ;
     803 }






    ./sysrap/storch.h:    unsigned matline ; 
    ./sysrap/storch.h:    printf("//storch::generate photon_id %3d genstep_id %3d  gs gentype/trackid/matline/numphoton(%3d %3d %3d %3d) type %d \n", 
    ./sysrap/storch.h:       gs.matline, 

    ./sysrap/SSim.cc:Lookup matline for bnd texture or array access 

    ./sysrap/sscint.h:    unsigned matline ; 
    ./sysrap/sscint.h:    gs.matline = 0u ;

    ./qudarap/qcerenkov.h:    //printf("//qcerenkov::wavelength_sampled_bndtex bnd %p gs.matline %d \n", bnd, gs.matline ); 
    ./qudarap/qcerenkov.h:        float4 props = bnd->boundary_lookup(wavelength, gs.matline, 0u); 
    ./qudarap/qcerenkov.h:    printf("//qcerenkov::wavelength_sampled_bndtex idx %6d sampledRI %7.3f cosTheta %7.3f sin2Theta %7.3f wavelength %7.3f count %d matline %d \n", 
    ./qudarap/qcerenkov.h:              idx , sampledRI, cosTheta, sin2Theta, wavelength, count, gs.matline );  

    ./qudarap/QDebug.cc:    unsigned cerenkov_matline = qb ? qb->qb->boundary_tex_MaterialLine_LS : 0 ;   
    ./qudarap/QDebug.cc:         << "AS NO QBnd at QDebug::MakeInstance the qdebug cerenkov genstep is using default matline of zero " << std::endl 
    ./qudarap/QDebug.cc:         << " cerenkov_matline " << cerenkov_matline  << std::endl
    ./qudarap/QDebug.cc:    scerenkov::FillGenstep( cerenkov_gs, cerenkov_matline, 100, dump ); 

    ./u4/U4.cc:    gs.matline = aMaterial->GetIndex() + SEvt::G4_INDEX_OFFSET ;  // offset signals that a mapping must be done in SEvt::setGenstep
    ./u4/U4.cc:    // note that gs.matline is not currently used for scintillation, 
    ./u4/U4.cc:    gs.matline = aMaterial->GetIndex() + SEvt::G4_INDEX_OFFSET ;  // offset signals that a mapping must be done in SEvt::setGenstep

    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ 



lookup_mtline
----------------

::


    171 /**
    172 SSim::lookup_mtline
    173 ---------------------
    174 
    175 Lookup matline for bnd texture or array access 
    176 from an original Geant4 material creation index
    177 as obtained by G4Material::GetIndex  
    178 
    179 NB this original mtindex is NOT GENERALLY THE SAME 
    180 as the Opticks material index. 
    181 
    182 **/
    183 
    184 int SSim::lookup_mtline( int mtindex ) const
    185 {
    186     return tree->lookup_mtline(mtindex);
    187 }


    epsilon:sysrap blyth$ opticks-f lookup_mtline
    ./CSG/CSGFoundry.h:    int lookup_mtline(int mtindex) const ; 
    ./CSG/CSGFoundry.cc:int CSGFoundry::lookup_mtline(int mtindex) const 
    ./CSG/CSGFoundry.cc:    return sim->lookup_mtline(mtindex) ;  
    ./sysrap/CheckGeo.cc:int CheckGeo::lookup_mtline(int mtindex) const 
    ./sysrap/stree.h:    int lookup_mtline( int mtindex ) const ; 
    ./sysrap/stree.h:inline int stree::lookup_mtline( int mtindex ) const 
    ./sysrap/CheckGeo.hh:    int                lookup_mtline(int mtindex) const ; 
    ./sysrap/tests/stree_material_test.cc:        int mtline = st.lookup_mtline(mtindex); 
    ./sysrap/tests/stree_material_test.cc:        int mtline = st.lookup_mtline(i); 
    ./sysrap/SSim.hh:    int lookup_mtline( int mtindex ) const ; 
    ./sysrap/SGeo.hh:        virtual int                lookup_mtline(int mtindex) const = 0 ; 
    ./sysrap/SEvt.cc:        int matline = cf ? cf->lookup_mtline(mtindex) : 0 ;
    ./sysrap/SSim.cc:SSim::lookup_mtline
    ./sysrap/SSim.cc:int SSim::lookup_mtline( int mtindex ) const
    ./sysrap/SSim.cc:    return tree->lookup_mtline(mtindex); 
    ./ggeo/GGeo.hh:        int  lookup_mtline(int mtindex) const ; 
    ./ggeo/GGeo.cc:int GGeo::lookup_mtline(int mtindex) const 
    epsilon:opticks blyth$ 




::


    2023-11-27 13:05:00.157 INFO  [436083] [SEvt::addGenstep@1893] SEvt::id EGPU (9)  GSV YES SEvt__OTHER
    2023-11-27 13:05:00.157 INFO  [436083] [SEvt::addGenstep@1922]  is_cerenkov_gs YES gentype 18 matline_ 1000001 G4_INDEX_OFFSET 1000000
    2023-11-27 13:05:00.157 INFO  [436083] [SEvt::addGenstep@1935]  is_cerenkov_gs YES cf YES gentype 18 mtindex 1 matline_ 1000001 matline -1

    2023-11-27 13:05:00.157 INFO  [436083] [SEvt::addGenstep@1893] SEvt::id ECPU (10)  GSV YES SEvt__OTHER
    2023-11-27 13:05:00.157 INFO  [436083] [SEvt::addGenstep@1922]  is_cerenkov_gs YES gentype 18 matline_ 4294967295 G4_INDEX_OFFSET 1000000
    2023-11-27 13:05:00.157 INFO  [436083] [SEvt::addGenstep@1935]  is_cerenkov_gs YES cf NO  gentype 18 mtindex 4293967295 matline_ 4294967295 matline 0

::

    In [1]: np.uint32(-1)                                                           
    Out[1]: 4294967295




    2023-11-27 13:05:00.226 INFO  [436083] [SEvt::addGenstep@1893] SEvt::id EGPU (9)  GSV YES SEvt__OTHER
    2023-11-27 13:05:00.226 INFO  [436083] [SEvt::addGenstep@1922]  is_cerenkov_gs YES gentype 18 matline_ 1000001 G4_INDEX_OFFSET 1000000
    2023-11-27 13:05:00.226 INFO  [436083] [SEvt::addGenstep@1935]  is_cerenkov_gs YES cf YES gentype 18 mtindex 1 matline_ 1000001 matline -1

    2023-11-27 13:05:00.226 INFO  [436083] [SEvt::addGenstep@1893] SEvt::id ECPU (10)  GSV YES SEvt__OTHER
    2023-11-27 13:05:00.226 INFO  [436083] [SEvt::addGenstep@1922]  is_cerenkov_gs YES gentype 18 matline_       4294967295 G4_INDEX_OFFSET 1000000
    2023-11-27 13:05:00.226 INFO  [436083] [SEvt::addGenstep@1935]  is_cerenkov_gs YES cf NO  gentype 18 mtindex 4293967295 matline_ 4294967295 matline 0
                                                                                                                    *



::

    356 void G4CXOpticks::init_SEvt()
    357 {
    358     sim->serialize() ;
    359     SEvt* sev = SEvt::CreateOrReuse(SEvt::EGPU) ;
    360 
    361     sev->setGeo((SGeo*)fd);    // Q: IS THIS USED BY ANYTHING ?  Y: Essential set_matline of Cerenkov Genstep 
    362 

::

    0360 int CSGFoundry::lookup_mtline(int mtindex) const
     361 {
     362     assert(sim);
     363     return sim->lookup_mtline(mtindex) ;
     364 }

    184 int SSim::lookup_mtline( int mtindex ) const
    185 {
    186     return tree->lookup_mtline(mtindex); 
    187 }    

    3517 inline int stree::lookup_mtline( int mtindex ) const
    3518 {
    3519     return mtindex_to_mtline.count(mtindex) == 0 ? -1 :  mtindex_to_mtline.at(mtindex) ;
    3520 }


::

    2243 inline void stree::import(const NPFold* fold)
    2244 {
    ....
    2269     NPFold* f_standard = fold->get_subfold(STANDARD) ;
    2270 
    2271     if(f_standard->is_empty())
    2272     {
    2273         std::cerr
    2274             << "stree::import skip asserts for empty f_standard : assuming trivial test geometry "
    2275             << std::endl
    2276             ;
    2277     }
    2278     else
    2279     {
    2280         standard->import(f_standard);
    2281 
    2282         assert( standard->bd );
    2283         NPX::VecFromArray<int4>( vbd, standard->bd );
    2284         standard->bd->get_names( bdname );
    2285 
    2286         assert( standard->bnd );
    2287         import_bnd( standard->bnd );
    2288     }

Looks like mtindex to mtline map only gets filled 
on import not on creation. That explains why things
work from a loaded geometry but not a created one::

    3458 /**
    3459 stree::import_bnd
    3460 -------------------
    3461 
    3462 Moved from SSim::import_bnd 
    3463 
    3464 **/
    3465 
    3466 inline void stree::import_bnd(const NP* bnd)
    3467 {
    3468     assert(bnd) ;
    3469     const std::vector<std::string>& bnames = bnd->names ;
    3470 
    3471     assert( mtline.size() == 0 );
    3472     assert( mtname.size() == mtindex.size() );
    3473 
    3474     // for each mtname use bnd->names to fill the mtline vector
    3475     SBnd::FillMaterialLine( mtline, mtindex, mtname, bnames );
    3476 
    3477     // fill (int,int) map from the mtline and mtindex vectors 
    3478     init_mtindex_to_mtline() ;
    3479 
    3480     if( level > 1 ) std::cerr
    3481         << "stree::import_bnd"
    3482         << " level > 1 [" << level << "]"
    3483         << " bnd " << bnd->sstr()
    3484         << " desc_mt "
    3485         << std::endl
    3486         << desc_mt()
    3487         << std::endl
    3488         ;
    3489 }







Suspect bug arises from the static::

    SEvt::AddGenstep(gs_) 

which adds to EGPU and ECPU and modifies its input, rather dirtily::

    1207 sgs SEvt::AddGenstep(const quad6& q)
    1208 {
    1209     sgs label = {} ;
    1210     if(Exists(0)) label = Get(0)->addGenstep(q) ;
    1211     if(Exists(1)) label = Get(1)->addGenstep(q) ;
    1212     return label ;
    1213 }



::

    279     quad6 gs_ = MakeGenstep_G4Cerenkov_modified( aTrack, aStep, numPhotons, betaInverse, pmin, pmax, maxCos, maxSin2, meanNumberOfPhotons1, meanNumberOfPhotons2 );
    280 
    281 #ifdef WITH_CUSTOM4
    282     sgs _gs = SEvt::AddGenstep(gs_);    // returns sgs struct which is a simple 4 int label 
    283     gs = C4GS::Make(_gs.index, _gs.photons, _gs.offset , _gs.gentype );
    284 #else
    285     gs = SEvt::AddGenstep(gs_);    // returns sgs struct which is a simple 4 int label 
    286 #endif
    287     // gs is primate static genstep label 
    288     // TODO: avoid the duplication betweek C and S with common SetGenstep private method
    289 
    290     if(dump) std::cout << "U4::CollectGenstep_G4Cerenkov_modified " << gs.desc() << std::endl ;
    291     LOG(LEVEL) << gs.desc();
    292 }





setGeo
---------

::

    epsilon:opticks blyth$ opticks-f setGeo\(
    ./CSG/tests/CSGFoundry_SGeo_SEvt_Test.cc:    sev->setGeo(fd); 
    ./sysrap/SEvt.hh:    void setGeo(const SGeo* cf); 
    ./sysrap/SEvt.cc:    sev->setGeo(fd);
    ./sysrap/SEvt.cc:void SEvt::setGeo(const SGeo* cf_)
    ./ggeo/GGeo.cc:    m_ok->setGeo((SGeo*)this);   //  for access to limited geometry info from lower levels 
    ./u4/tests/U4HitTest.cc:    sev->setGeo(fd); 
    ./optickscore/Opticks.hh:       void        setGeo( const SGeo* geo ); 
    ./optickscore/Opticks.cc:void Opticks::setGeo(const SGeo* geo)
    ./examples/UseOptiX7GeometryInstancedGASCompDyn/SBT.h:    void setGeo(const Geo* geo); 
    ./examples/UseOptiX7GeometryInstancedGASCompDyn/SBT.cc:void SBT::setGeo(const Geo* geo)
    ./examples/UseOptiX7GeometryInstancedGASCompDyn/UseOptiX7GeometryInstancedGASCompDyn.cc:    sbt.setGeo(&geo); 
    ./g4cx/G4CXOpticks.cc:    sev->setGeo((SGeo*)fd);    // Q: IS THIS USED BY ANYTHING ?  Y: Essential set_matline of Cerenkov Genstep 
    epsilon:opticks blyth$ 



TODO : review U4Tree geometry transition to see how to form the mtindex to mtline map on production side
------------------------------------------------------------------------------------------------------------




