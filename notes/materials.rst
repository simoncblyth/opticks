Materials 
===========

References
------------

Optical Scattering Lengths in Large Liquid-Scintillator Neutrino Detectors, M.Wurm et.al.

* http://arxiv.org/pdf/1004.0811.pdf 

Neutrino Physics With JUNO p19

Require: Attenuation Length of LS greater than 20m at 430 nm, which corresponds to an 
absorption length of 60m with a Rayleigh scattering length of 30m.
Rayleigh scattering length of LAB measured 28.2+-1.0 m at 430nm (Liu Q, NIM)

::

    1/60 + 1/30 = 3/60 = 1/20       

                        @ 430 nm

    Attenuation Length ~ 20m

    Scattering Length  ~ 30m    << scattering dominant over absorption in scintillator
    Absorption Length  ~ 60m

Absorption length is difficult to measure, instead the overall attenuation length
and Rayleigh scattering length are measured and the absorption length
inferred from those.

Rayleigh scattering and depolarization ratio in linear alkylbenzene

* http://arxiv.org/abs/1504.01001
* http://arxiv.org/pdf/1504.01001v2.pdf


Attenuation Length of Water
-----------------------------

* http://www.inp.demokritos.gr/nestor/2nd/files/247_252_bradner.pdf

Clear water has a sharp optical transparency window in the blue-green, 
ie attenuation rapidly increases (attenuation length decreases) 
outside wavelength range 400-570nm (3-2.2 eV)

...it is common to assume that the scattering
contributes about 20% to the total attenuation coefficient in clear ocean water.

Absorption dominates over scattering in water

::

     1/At = 1/Sc + 1/Ab

::

    In [6]: w = np.linspace(60, 820, 20)

    In [7]: e = 1240./w   ##   hc ~ 1240 nm.eV

    In [8]: np.dstack([w, e])
    Out[8]: 
    array([[[  60.        ,   20.66666667],
            [ 100.        ,   12.4       ],
            [ 140.        ,    8.85714286],
            [ 180.        ,    6.88888889],
            [ 220.        ,    5.63636364],
            [ 260.        ,    4.76923077],
            [ 300.        ,    4.13333333],
            [ 340.        ,    3.64705882],
            [ 380.        ,    3.26315789],
            [ 420.        ,    2.95238095],
            [ 460.        ,    2.69565217],
            [ 500.        ,    2.48      ],
            [ 540.        ,    2.2962963 ],
            [ 580.        ,    2.13793103],
            [ 620.        ,    2.        ],
            [ 660.        ,    1.87878788],
            [ 700.        ,    1.77142857],
            [ 740.        ,    1.67567568],
            [ 780.        ,    1.58974359],
            [ 820.        ,    1.51219512]]])



Rayleigh Scattering
----------------------

::

    intensity ~ wavelength^-4   

    # blue(400nm) scatters ~10 times more than red(700nm) -> color of sky 

::

    In [13]: np.power([400.,500.,600.,700.,800.], -4)*1e12
    Out[13]: array([ 39.0625,  16.    ,   7.716 ,   4.1649,   2.4414])

    In [14]: np.power([400.,500.,600.,700.,800.], -4)*1e12/2.4414
    Out[14]: array([ 16.    ,   6.5536,   3.1605,   1.706 ,   1.    ])

    In [15]: np.power([400.,500.,600.,700.,800.], -4)*1e12/39.0625
    Out[15]: array([ 1.    ,  0.4096,  0.1975,  0.1066,  0.0625])


* so scattering length greater at higher wavelengths, as less scattering


Color of Water
----------------

* https://en.wikipedia.org/wiki/Color_of_water

While relatively small quantities of water appear to be colorless, water's tint
becomes a deeper blue as the thickness of the observed sample increases. The
blue hue of water is an intrinsic property and is caused by the selective
absorption and scattering of white light. Impurities dissolved or suspended in
water may give water different colored appearances.


::

    In [1]: np.dstack([w, a[0,:,0], a[0,:,1],a[0,:,2],a[0,:,3],a[1,:,0]])   # Water 
    Out[1]: 
                                         // absorption     scattering 
                                         // length         length
                                         // (mm)           (mm)
                                           
    array([[[      60.    ,        1.39  ,      273.208 ,  1000000.    ,        0.    ,      300.    ],
            [      80.    ,        1.39  ,      273.208 ,  1000000.    ,        0.    ,      300.    ],
            [     100.    ,        1.39  ,      273.208 ,  1000000.    ,        0.    ,      300.    ],
            [     120.    ,        1.39  ,      273.208 ,  1000000.    ,        0.    ,      300.    ],
            [     140.    ,        1.39  ,      273.208 ,  1000000.    ,        0.    ,      300.    ],
            [     160.    ,        1.39  ,      273.208 ,  1000000.    ,        0.    ,      300.    ],
            [     180.    ,        1.39  ,      273.208 ,  1000000.    ,        0.    ,      300.    ],
            [     200.    ,        1.39  ,      691.5562,  1000000.    ,        0.    ,      300.    ],
            [     220.    ,        1.3841,     1507.1183,  1000000.    ,        0.    ,      300.    ],
            [     240.    ,        1.3783,     2228.2798,  1000000.    ,        0.    ,      300.    ],
            [     260.    ,        1.3724,     3164.6375,  1000000.    ,        0.    ,      300.    ],
            [     280.    ,        1.3666,     4286.0454,  1000000.    ,        0.    ,      300.    ],
            [     300.    ,        1.3608,     5992.6128,  1000000.    ,        0.    ,      300.    ],
            [     320.    ,        1.3595,     7703.5034,  1000000.    ,        0.    ,      300.    ],
            [     340.    ,        1.3585,    10257.2852,  1000000.    ,        0.    ,      300.    ],
            [     360.    ,        1.3572,    12811.0684,  1000000.    ,        0.    ,      300.    ],
            //
            //  blue : larger absorption length, less absorption -> water looks blue
            //
            [     380.    ,        1.356 ,    15364.8496,  1000000.    ,        0.    ,      300.    ],
            [     400.    ,        1.355 ,    19848.9316,  1000000.    ,        0.    ,      300.    ],
            [     420.    ,        1.354 ,    24670.9512,  1000000.    ,        0.    ,      300.    ],
            [     440.    ,        1.353 ,    27599.9746,  1000000.    ,        0.    ,      300.    ],
            [     460.    ,        1.3518,    28732.2051,  1000000.    ,        0.    ,      300.    ],
            [     480.    ,        1.3505,    29587.0527,  1000000.    ,        0.    ,      300.    ],
            [     500.    ,        1.3492,    26096.2637,  1000000.    ,        0.    ,      300.    ],
            [     520.    ,        1.348 ,    17787.9492,  1000000.    ,        0.    ,      300.    ],
            [     540.    ,        1.347 ,    16509.3672,  1000000.    ,        0.    ,      300.    ],
            [     560.    ,        1.346 ,    13644.791 ,  1000000.    ,        0.    ,      300.    ],
            [     580.    ,        1.345 ,    10050.459 ,  1000000.    ,        0.    ,      300.    ],
            [     600.    ,        1.344 ,     4328.5166,  1000000.    ,        0.    ,      300.    ],
            [     620.    ,        1.3429,     3532.6135,  1000000.    ,        0.    ,      300.    ],
            [     640.    ,        1.3419,     3149.8655,  1000000.    ,        0.    ,      300.    ],
            [     660.    ,        1.3408,     2404.4004,  1000000.    ,        0.    ,      300.    ],
            [     680.    ,        1.3397,     2126.562 ,  1000000.    ,        0.    ,      300.    ],
            //
            //  red :  smaller absorption length, ie more absorption
            //
            [     700.    ,        1.3387,     1590.72  ,  1000000.    ,        0.    ,      300.    ],
            [     720.    ,        1.3376,      809.6543,  1000000.    ,        0.    ,      300.    ],
            [     740.    ,        1.3365,      370.1322,  1000000.    ,        0.    ,      300.    ],
            [     760.    ,        1.3354,      371.9737,  1000000.    ,        0.    ,      300.    ],
            [     780.    ,        1.3344,      425.7059,  1000000.    ,        0.    ,      300.    ],
            [     800.    ,        1.3333,      486.681 ,  1000000.    ,        0.    ,      300.    ],
            [     820.    ,        1.3333,      486.681 ,  1000000.    ,        0.    ,      300.    ]]])



Rayleigh Scattering Accounting for Polarization
-------------------------------------------------

* http://www.philiplaven.com/p8b.html


Property Domain Sanity Check
------------------------------

Material properties are written by G4DAE into extra elements
of the COLLADA export. Below shows the domain is written in Geant4 
native energy and property values.
Assuming Geant4/CLHEP native units (to be confirmed):

* energy in units of keV ?
* scattering/absorption lengths in mm ?

G4DAEWrite.cc::

    407 void G4DAEWrite::PropertyVectorWrite(const G4String& key,
    408                            const G4MaterialPropertyVector* const pvec,
    409                             xercesc::DOMElement* extraElement)
    410 {
    411 
    412    std::ostringstream pvalues;
    413 
    414 #ifdef _GEANT4_TMP_GEANT94_
    415    for (G4int i=0; i<pvec->Entries(); i++)
    416    {
    417      G4MPVEntry cval = pvec->GetEntry(i);
    418      if (i!=0)  { pvalues << " "; }
    419      pvalues << cval.GetPhotonEnergy() << " " << cval.GetProperty();
    420    }
    421 #else
    422    for (size_t i=0; i<pvec->GetVectorLength(); i++)
    423    {
    424        if (i!=0)  { pvalues << " "; }
    425        pvalues << pvec->Energy(i) << " " << (*pvec)[i];
    426    }
    427 #endif
    428 

AssimpGGeo.cc::

     466 
     467             //printf("AssimpGGeo::convertMaterials aiScene materialIndex %u (GMaterial) name %s \n", i, name);
     468             GMaterial* gmat = new GMaterial(name, index);
     469             gmat->setStandardDomain(standard_domain);
     470             addProperties(gmat, mat );
     471             gg->add(gmat);
     472 
     473             {
     474                 // without standard domain applied
     475                 GMaterial* gmat_raw = new GMaterial(name, index);
     476                 addProperties(gmat_raw, mat );
     477                 gg->addRaw(gmat_raw);
     478             }


Domain scaling and taking reciprocal is done in the construction of opticks GMaterial 
done in AssimpGGeo::addPropertyVector converting into nanometer.
Also for non-raw materials a domain standardization is applied, such that 
all values are available at specific wavelengths::

     189     // dont scale placeholder -1 : 1 domain ranges
     190     double dscale = data[0] > 0 && data[npair-1] > 0 ? m_domain_scale : 1.f ;
     191     double vscale = m_values_scale ;
     192 
     ...
     214     std::vector<float> vals ;
     215     std::vector<float> domain  ;
     216 
     217     for( unsigned int i = 0 ; i < npair ; i++ )
     218     {
     219         double d0 = data[2*i] ;
     220         double d = m_domain_reciprocal ? dscale/d0 : dscale*d0 ;
     221         double v = data[2*i+1]*vscale  ;
     222 
     223         double dd = noscale ? d0 : d ;
     224 
     225         domain.push_back( static_cast<float>(dd) );
     226         vals.push_back( static_cast<float>(v) );
     227 
     228         //if( noscale && ( i < 5 || i > npair - 5) )
     229         //printf("%4d %10.3e %10.3e \n", i, domain.back(), vals.back() );
     230     }

     068 AssimpGGeo::AssimpGGeo(GGeo* ggeo, AssimpTree* tree, AssimpSelection* selection)
      69    :
      70    m_ggeo(ggeo),
      71    m_tree(tree),
      72    m_selection(selection),
      73    m_domain_scale(1.f),
      74    m_values_scale(1.f),
      75    m_domain_reciprocal(true),
      76    m_skin_surface(0),
     ...

     100 void AssimpGGeo::init()
     101 {
     102     // TODO: consolidate constant handling into okc-
     103     //       see also ggeo-/GConstant and probably elsewhere
     104     //
     105     // see g4daenode.py as_optical_property_vector
     106 
     107     double hc_over_GeV = 1.2398424468024265e-06 ;  // h_Planck * c_light / GeV / nanometer #  (approx, hc = 1240 eV.nm )  
     108     double hc_over_MeV = hc_over_GeV*1000. ;
     109     //float hc_over_eV  = hc_over_GeV*1.e9 ;
     110 
     111     m_domain_scale = static_cast<float>(hc_over_MeV) ;
     112     m_values_scale = 1.0f ;
     113 
     114     m_volnames = m_ggeo->isVolnames();
     115 }

