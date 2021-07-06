wavelength_cfplot_shows_zeros_in_2nm_bins_in_350nm_range FIXED
=================================================================

FIXED 
-------

* this was a problem with quadrap QCtx/QTex the GPU filtermode was Point not Linear so the 
  GPU interpolation was not being done...

* initially I did not realise that and devised the multiresolution tex approach to effectively 20x the bins
  in the tail while only pay 3x 

* switching on the interpolation avoiding the severe initial problem, but the multi-resolution approach 
  still improves things futher reducing wavelength_cfplot.py chi2 


issue
--------

Looks like some technical artifact clumping up the bins::

    In [22]: np.histogram( wl.w[0], bins=np.arange(300,400,2) )                                                                                                                           
    Out[22]: 
    (array([    0,     0,   234,     0,     0,   229,     0,     0,     0,   242,     0,     0,     0,   258,     0,     0,   268,     0,   270,     0,   249,     0,   230,     0,   253,     0,   247,
              214,   242,   237,   235,   497,   495,   267,   473,   509,   478,   697,   726,  1194,  1906,  2575,  4143,  6379, 10177, 14655, 19673, 24751, 28793]),
     array([300, 302, 304, 306, 308, 310, 312, 314, 316, 318, 320, 322, 324, 326, 328, 330, 332, 334, 336, 338, 340, 342, 344, 346, 348, 350, 352, 354, 356, 358, 360, 362, 364, 366, 368, 370, 372, 374,
            376, 378, 380, 382, 384, 386, 388, 390, 392, 394, 396, 398]))

    In [23]:                                                                                                                                                                              

    In [23]: np.histogram( wl.w[1], bins=np.arange(300,400,2) )                                                                                                                           
    Out[23]: 
    (array([   85,    67,    72,    57,    59,    66,    68,    59,    74,    68,    64,    65,    75,    68,    59,   113,   112,   115,    94,   110,   109,   114,   120,   114,   111,   154,   178,
              203,   257,   282,   346,   390,   363,   358,   416,   526,   485,   701,   898,  1215,  1756,  2689,  4247,  6534,  9917, 14570, 19667, 24899, 28551]),
     array([300, 302, 304, 306, 308, 310, 312, 314, 316, 318, 320, 322, 324, 326, 328, 330, 332, 334, 336, 338, 340, 342, 344, 346, 348, 350, 352, 354, 356, 358, 360, 362, 364, 366, 368, 370, 372, 374,
            376, 378, 380, 382, 384, 386, 388, 390, 392, 394, 396, 398]))



What are the source energies and wavelengths ?


Are the raw (non-standardized) materials persisted ?
--------------------------------------------------------

::

    093 /**
     94 X4MaterialTable::init
     95 -----------------------
     96 
     97 For all materials obtained from G4Material::GetMaterialTable() apply X4Material::Convert
     98 collecting in standardized and raw forms into m_mlib.
     99 
    100 G4Material which have an EFFICIENCY property are collected into m_material_with_efficiency vector.
    101 
    102 **/
    103 
    104 void X4MaterialTable::init()
    105 {
    106     unsigned num_input_materials = m_input_materials.size() ;
    107 
    108     LOG(LEVEL) << ". G4 nmat " << num_input_materials ;
    109 
    110     for(unsigned i=0 ; i < num_input_materials ; i++)
    111     {
    112         G4Material* material = m_input_materials[i] ;
    ...
    126         char mode_g4interpolated = 'G' ;
    127         //char mode_oldstandardized = 'S' ;
    128         char mode_asis = 'A' ;
    129 
    130         GMaterial* mat = X4Material::Convert( material, mode_g4interpolated );
    131         if(mat->hasProperty("EFFICIENCY"))
    132         {
    133              m_materials_with_efficiency.push_back(material);
    134         }
    135         m_mlib->add(mat) ;
    136 
    137 
    138         GMaterial* rawmat = X4Material::Convert( material, mode_asis );
    139         m_mlib->addRaw(rawmat) ;
    140     }
    141 }

    0328 void GMaterialLib::addRaw(GMaterial* mat)
     329 {
     330     m_materials_raw.push_back(mat);
     331 }


* the GMaterialLib does not persist the raw materials
* BUT GScintillatorLib does  
 

::


    1122 std::vector<GMaterial*> GGeo::getRawMaterialsWithProperties(const char* props, char delim) const
    1123 {
    1124     return m_materiallib->getRawMaterialsWithProperties(props, delim );
    1125 }

    1075 std::vector<GMaterial*> GMaterialLib::getRawMaterialsWithProperties(const char* props, char delim) const
    1076 {
    1077     std::vector<std::string> elem ;
    1078     BStr::split(elem, props, delim);
    1079 
    1080     LOG(LEVEL)
    1081          << props
    1082          << " m_materials_raw.size()  " << m_materials_raw.size()
    1083          ;
    1084 
    1085     std::vector<GMaterial*>  selected ;
    1086     for(unsigned int i=0 ; i < m_materials_raw.size() ; i++)
    1087     {
    1088         GMaterial* mat = m_materials_raw[i];
    1089         unsigned int found(0);
    1090         for(unsigned int p=0 ; p < elem.size() ; p++)
    1091         {
    1092            if(mat->hasProperty(elem[p].c_str())) found+=1 ;
    1093         }
    1094         if(found == elem.size()) selected.push_back(mat);
    1095     }
    1096     return selected ;
    1097 }

    ...
    1240 void GGeo::prepareScintillatorLib()
    1241 {   
    1242     LOG(verbose) << "GGeo::prepareScintillatorLib " ;
    1243     
    1244     findScintillatorMaterials("SLOWCOMPONENT,FASTCOMPONENT,REEMISSIONPROB");
    1245     
    1246     unsigned int nscint = getNumScintillatorMaterials() ;
    1247     
    1248     if(nscint == 0)
    1249     {   
    1250         LOG(LEVEL) << " found no scintillator materials  " ;
    1251     }
    1252     else
    1253     {   
    1254         LOG(LEVEL) << " found " << nscint << " scintillator materials  " ;
    1255         
    1256         GScintillatorLib* sclib = getScintillatorLib() ;
    1257         
    1258         for(unsigned int i=0 ; i < nscint ; i++)
    1259         {   
    1260             GPropertyMap<double>* scint = dynamic_cast<GPropertyMap<double>*>(getScintillatorMaterial(i));
    1261             sclib->add(scint);
    1262         }
    1263         
    1264         sclib->close();
    1265     }
    1266 }


    099 void GScintillatorLib::add(GPropertyMap<double>* scint)
    100 {
    101     assert(!isClosed());
    102     addRaw(scint);
    103 }

    774 void GPropertyLib::addRaw(GPropertyMap<double>* pmap)
    775 {
    776     m_raw.push_back(pmap);
    777 }

    834 void GPropertyLib::saveRaw()
    835 {
    836     std::string dir = getCacheDir();
    837     unsigned int nraw = m_raw.size();
    838     for(unsigned int i=0 ; i < nraw ; i++)
    839     {
    840         GPropertyMap<double>* pmap = m_raw[i] ;
    841         pmap->save(dir.c_str());
    842     }
    843 }


    1267 
    1268 void GGeo::findScintillatorMaterials(const char* props)
    1269 {
    1270     m_scintillators_raw = getRawMaterialsWithProperties(props, ',');
    1271 }   
    1272 
    1273 void GGeo::dumpScintillatorMaterials(const char* msg)
    1274 {
    1275     LOG(info)<< msg ;
    1276     for(unsigned int i=0; i<m_scintillators_raw.size() ; i++)
    1277     {
    1278         GMaterial* mat = m_scintillators_raw[i];
    1279         //mat->Summary();
    1280         std::cout << std::setw(30) << mat->getShortName()
    1281                   << " keys: " << mat->getKeysString()
    1282                   << std::endl ;  
    1283     }              
    1284 }   



GPU texture formed from the icdf in the GScintillatorLib buffer::

     19 QScint::QScint(const GScintillatorLib* slib_)
     20     :
     21     slib(slib_),
     22     dsrc(slib->getBuffer()),
     23     src(NPY<double>::MakeFloat(dsrc)),
     24     tex(nullptr)
     25 {
     26     INSTANCE = this ;
     27     init();
     28 }
     29 
     30 void QScint::init()
     31 {
     32     makeScintTex(src) ;
     33 }
     34 



In [1]: fs = np.load("SLOWCOMPONENT.npy")                                                                                                                                             



Hmm energy to wavelength conversion with 1240 might avoid the .999 .001 glitches::

    In [2]: fs                                                                                                                                                                            
    Out[2]: 
    array([[ 79.99 ,   0.   ],
           [120.023,   0.   ],
           [199.974,   0.   ],
           [330.   ,   0.002],
           [331.   ,   0.002],
           [332.   ,   0.002],
           [333.   ,   0.002],
           [334.   ,   0.002],
           [335.   ,   0.002],
           [336.   ,   0.002],
           [337.   ,   0.002],
           [338.   ,   0.002],
           [339.   ,   0.002],
           [340.   ,   0.002],
           [340.999,   0.002],
           [342.   ,   0.002],
           [343.   ,   0.002],
           [344.   ,   0.003],
           [344.999,   0.002],
           [346.   ,   0.002],
           [347.   ,   0.003],
           [348.   ,   0.002],
           [349.   ,   0.003],
           [350.   ,   0.003],
           [351.   ,   0.003],
           [352.   ,   0.003],
           [353.   ,   0.003],
           [354.   ,   0.004],
           [355.001,   0.004],
           [356.   ,   0.005],
           [357.   ,   0.005],
           [358.   ,   0.006],
           [359.   ,   0.006],
           [359.999,   0.007],
           [361.001,   0.007],
           [362.   ,   0.008],
           [363.   ,   0.008],
           [364.   ,   0.009],
           [365.   ,   0.009],
           [366.   ,   0.008],
           [367.   ,   0.009],
           [368.   ,   0.009],
           [368.999,   0.009],
           [370.   ,   0.01 ],
           [371.   ,   0.011],



Checks
------


ana/ScintillationIntegral.py
    plots icdf

ana/wavelength.py 
    loads samples

ana/wavelength_plt.py 
    demo plot showing distrib together with material props

ana/wavelength_cfplot.py 
    chi2 comparison

    comparing tex sampling and G4 sampling gives good agreement 
    if the extremes are excluded:: 

        dom = np.arange(385, 475, 1)  

    the problem is an exceedingly steep icdf at the extremes  


qudarap/tests/QCtxTest.cc
    GPU tex samples 




Workaround with tex in triplicate using log probabilities ?
-------------------------------------------------------------

* could provide tex in triplicate 

1. ln(prob) for u < 0.1
2. prob for u 0.1:0.9
3. ln(1-prob) for u > 0.9

* hmm using log probabilitis is kinda confusing and needs converting around

    In [25]: a = np.linspace(0,0.1,100)                                                                                                                                                   

    In [26]: a                                                                                                                                                                            
    Out[26]: 
    array([0.    , 0.001 , 0.002 , 0.003 , 0.004 , 0.0051, 0.0061, 0.0071, 0.0081, 0.0091, 0.0101, 0.0111, 0.0121, 0.0131, 0.0141, 0.0152, 0.0162, 0.0172, 0.0182, 0.0192, 0.0202, 0.0212, 0.0222, 0.0232,
           0.0242, 0.0253, 0.0263, 0.0273, 0.0283, 0.0293, 0.0303, 0.0313, 0.0323, 0.0333, 0.0343, 0.0354, 0.0364, 0.0374, 0.0384, 0.0394, 0.0404, 0.0414, 0.0424, 0.0434, 0.0444, 0.0455, 0.0465, 0.0475,
           0.0485, 0.0495, 0.0505, 0.0515, 0.0525, 0.0535, 0.0545, 0.0556, 0.0566, 0.0576, 0.0586, 0.0596, 0.0606, 0.0616, 0.0626, 0.0636, 0.0646, 0.0657, 0.0667, 0.0677, 0.0687, 0.0697, 0.0707, 0.0717,
           0.0727, 0.0737, 0.0747, 0.0758, 0.0768, 0.0778, 0.0788, 0.0798, 0.0808, 0.0818, 0.0828, 0.0838, 0.0848, 0.0859, 0.0869, 0.0879, 0.0889, 0.0899, 0.0909, 0.0919, 0.0929, 0.0939, 0.0949, 0.096 ,
           0.097 , 0.098 , 0.099 , 0.1   ])

    In [27]: np.log(a)                                                                                                                                                                    
    /Users/blyth/miniconda3/bin/ipython:1: RuntimeWarning: divide by zero encountered in log
      #!/Users/blyth/miniconda3/bin/python
    Out[27]: 
    array([   -inf, -6.8977, -6.2046, -5.7991, -5.5114, -5.2883, -5.1059, -4.9518, -4.8183, -4.7005, -4.5951, -4.4998, -4.4128, -4.3328, -4.2586, -4.1897, -4.1251, -4.0645, -4.0073, -3.9533, -3.902 ,
           -3.8532, -3.8067, -3.7622, -3.7197, -3.6788, -3.6396, -3.6019, -3.5655, -3.5304, -3.4965, -3.4637, -3.432 , -3.4012, -3.3713, -3.3424, -3.3142, -3.2868, -3.2601, -3.2341, -3.2088, -3.1841,
           -3.16  , -3.1365, -3.1135, -3.091 , -3.0691, -3.0476, -3.0265, -3.0059, -2.9857, -2.9659, -2.9465, -2.9274, -2.9087, -2.8904, -2.8724, -2.8547, -2.8373, -2.8202, -2.8034, -2.7868, -2.7706,
           -2.7546, -2.7388, -2.7233, -2.7081, -2.693 , -2.6782, -2.6636, -2.6492, -2.635 , -2.621 , -2.6072, -2.5936, -2.5802, -2.567 , -2.5539, -2.541 , -2.5283, -2.5157, -2.5033, -2.491 , -2.4789,
           -2.4669, -2.4551, -2.4434, -2.4318, -2.4204, -2.4091, -2.3979, -2.3868, -2.3759, -2.3651, -2.3544, -2.3438, -2.3334, -2.323 , -2.3127, -2.3026])



Basic problem is not enough bins of GPU texture to do justice in the extremes
--------------------------------------------------------------------------------

So x10 the sampling for n < 0.1 and u > 0.9 : then can just use linear mapping.

::

    078 /**
     79 qctx::scint_wavelength_with_fine_binned_extremes
     80 --------------------------------------------------
     81 
     82 Idea is to improve handling of extremes by throwing ten times the bins
     83 at those regions, using simple and cheap linear mappings.
     84 
     85 Perhaps could also use log probabilities to do something similar to 
     86 this in a fancy way : just like using log scale to give more detail in the low registers. 
     87 But that has computational disadvantage of expensive mapping functions to get between spaces. 
     88 
     89 **/
     90 
     91 inline QCTX_METHOD float qctx::scint_wavelength_with_fine_binned_extremes(curandStateXORWOW& rng)
     92 {
     93     float u0 = curand_uniform(&rng);
     94 
     95     float wl ;
     96     if( u0 < 0.1f )
     97     {
     98         wl = tex2D<float>(scint_tex, u0*10.f , 1.f );
     99     }
    100     else if ( u0 > 0.9f )
    101     {
    102         wl = tex2D<float>(scint_tex, (u0 - 0.9f)*10.f , 2.f );
    103     }
    104     else
    105     {
    106         wl = tex2D<float>(scint_tex, u0,  0.f );
    107     }
    108     return wl ;
    109 }



Can avoid the below by using Geant4 G4PhysicsVector::GetEnergy interpolation

::

    204 GProperty<double>* GScintillatorLib::constructInvertedReemissionCDF(GPropertyMap<double>* pmap)
    205 {
    206     std::string name = pmap->getShortNameString();
    207 
    208     typedef GProperty<double> P ;
    209 
    210     P* slow = getProperty(pmap, slow_component);
    211     P* fast = getProperty(pmap, fast_component);
    212     assert(slow != NULL && fast != NULL );
    213 
    214 
    215     double mxdiff = GProperty<double>::maxdiff(slow, fast);
    216     assert(mxdiff < 1e-6 );
    217 
    218     P* rrd = slow->createReversedReciprocalDomain();    // have to used reciprocal "energywise" domain for G4/NuWa agreement
    219 
    220     P* srrd = rrd->createZeroTrimmed();                 // trim extraneous zero values, leaving at most one zero at either extremity
    221 
    222     unsigned int l_srrd = srrd->getLength() ;
    223     unsigned int l_rrd = rrd->getLength()  ;
    224 
    225     if( l_srrd != l_rrd - 2)
    226     {
    227        LOG(debug)
    228            << "was expecting to trim 2 values "
    229            << " l_srrd " << l_srrd
    230            << " l_rrd " << l_rrd
    231            ;
    232     }
    233     //assert( l_srrd == l_rrd - 2); // expect to trim 2 values
    234 
    235     P* rcdf = srrd->createCDF();
    236 
    237     P* icdf = rcdf->createInverseCDF(m_icdf_length);
    238 
    239     icdf->getValues()->reciprocate();  // avoid having to reciprocate lookup results : by doing it here 
    240 
    241     return icdf ;
    242 }



::


     79 void G4PhysicsOrderedFreeVector::InsertValues(G4double energy, G4double value)
     80 { 
     81         std::vector<G4double>::iterator binLoc =
     82                  std::lower_bound(binVector.begin(), binVector.end(), energy);
 
     ///   Returns an iterator pointing to the first element in the range [first, last)
     ///   that is not less than (i.e. greater or equal to) value, or last if no such
     ///   element is found.  

     83  
     84         size_t binIdx = binLoc - binVector.begin(); // Iterator difference!
     85    
     86         std::vector<G4double>::iterator dataLoc = dataVector.begin() + binIdx;
     87  
     88         binVector.insert(binLoc, energy);
     89         dataVector.insert(dataLoc, value);
     90  
     91         ++numberOfNodes;
     92         edgeMin = binVector.front();
     93         edgeMax = binVector.back();
     94 } 

     




