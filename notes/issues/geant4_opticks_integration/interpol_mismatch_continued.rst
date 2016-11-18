Interpolation Mismatch Continued
===================================

Issue
------

Whilst working on tconcentric distrib chi2 the time distrib stood out as 
large chi2 contributor. 

Although interpol difference is not  
particularly large for GROUPVEL (up to 1.2%)  the simulation time 
is obtained directly from the ratio of propagated distance to GROUPVEL 
so interpol differences directly feed into the T distrib chi2.

For other properties small interpol differences make little
impact as they are "protected" by random number throws for
example converting absorption and scattering lengths into 
absorption and scattering distances. 


TODO: check time distrib diff matches expectation from groupvel interpol mismatch
----------------------------------------------------------------------------------


Testing finebndtex
-------------------

::

    lldb CInterpolationTest -- --finebndtex
    lldb OInterpolationTest -- --finebndtex


How to up samples ?
--------------------

Material buf is now by default recreated postcache, in order to replaceGROUPVEL, 
so upsampling can happen there::

    052 void GMaterialLib::postLoadFromCache()
    053 {
    ...
    116     if(groupvel)
    117     {
    118        bool debug = true ;
    119        replaceGROUPVEL(debug);
    120     }
    121 
    122     if(nore || noab || nosc || xxre || xxab || xxsc || fxre || fxsc || fxab || groupvel)
    123     {
    124         // need to replace the loaded buffer with a new one with the changes for Opticks to see it 
    125         NPY<float>* mbuf = createBuffer();
    126         setBuffer(mbuf);
    127     }
    128 
    129 }
    130 
    ...
    293 NPY<float>* GMaterialLib::createBuffer()
    294 {
    295     return createBufferForTex2d() ;
    296 }
    297 
    298 
    299 NPY<float>* GMaterialLib::createBufferForTex2d()
    300 {
    301     // trying to arrange the memory layout of this buffer to 
    302     // match the requirements of tex2d<float4>
    303 
    304     unsigned int ni = getNumMaterials();
    305     unsigned int nj = NUM_FLOAT4 ;
    306     unsigned int nk = getStandardDomain()->getLength();
    307     unsigned int nl = 4 ;
    308 
    309 
    310     if(ni == 0 || nj == 0)
    311     {
    312         LOG(error) << "GMaterialLib::createBufferForTex2d"
    313                    << " NO MATERIALS ? "
    314                    << " ni " << ni
    315                    << " nj " << nj
    316                    ;
    317 
    318         return NULL ;
    319     }
    320 
    321 
    322 
    323     NPY<float>* mbuf = NPY<float>::make(ni, nj, nk, nl);  // materials/payload-category/wavelength-samples/4prop
    324     mbuf->zero();
    325     float* data = mbuf->getValues();
    326 
    327     for(unsigned int i=0 ; i < ni ; i++)
    328     {
    329         GMaterial* mat = m_materials[i] ;
    330         GProperty<float> *p0,*p1,*p2,*p3 ;
    331 
    332         for(unsigned int j=0 ; j < nj ; j++)
    333         {
    334             p0 = mat->getPropertyByIndex(j*4+0);
    335             p1 = mat->getPropertyByIndex(j*4+1);
    336             p2 = mat->getPropertyByIndex(j*4+2);
    337             p3 = mat->getPropertyByIndex(j*4+3);
    338 
    339             for( unsigned int k = 0; k < nk; k++ )    // over wavelength-samples
    340             {
    341                 unsigned int offset = i*nj*nk*nl + j*nk*nl + k*nl ;
    342 
    343                 data[offset+0] = p0 ? p0->getValue(k) : MATERIAL_UNSET ;
    344                 data[offset+1] = p1 ? p1->getValue(k) : MATERIAL_UNSET ;


::
 
     27 void GMaterial::init()
     28 {   
     29     GDomain<float>* sd = GPropertyLib::getDefaultDomain();
     30     setStandardDomain(sd);
     31 }   


     62 GDomain<float>* GPropertyLib::getDefaultDomain()
     63 {
     64    return new GDomain<float>(Opticks::DOMAIN_LOW, Opticks::DOMAIN_HIGH, Opticks::DOMAIN_STEP );
     65 }



Where is GGeo standardization interpolation done::

     757 template <typename T>
     758 GProperty<T>* GProperty<T>::createInterpolatedProperty(GDomain<T>* domain)
     759 {
     760     GAry<T>* idom = new GAry<T>(domain->getLength(), domain->getValues());
     761     GAry<T>* ival = GAry<T>::np_interp( idom , m_domain, m_values );
     762 
     763     GProperty<T>* prop = new GProperty<T>( ival, idom );
     764     return prop ;
     765 }
     766 
     767 template <typename T>
     768 T GProperty<T>::getInterpolatedValue(T x)
     769 {
     770     // find the value "y" at "x" by first placing "x" within the domain
     771     // and then using linear interpolation of the above and below values
     772     return GAry<T>::np_interp( x , m_domain, m_values );
     773 }
::

    simon:ggeo blyth$ opticks-find createInterpolated 
    ./ggeo/GProperty.cc:GProperty<T>* GProperty<T>::createInterpolatedProperty(GDomain<T>* domain)
    ./ggeo/GPropertyMap.cc:       GProperty<T>* ipol = orig->createInterpolatedProperty(m_standard_domain); 
    ./ggeo/GProperty.hh:   GProperty<T>* createInterpolatedProperty(GDomain<T>* domain);
    simon:opticks blyth$ 



::

     467             //printf("AssimpGGeo::convertMaterials aiScene materialIndex %u (GMaterial) name %s \n", i, name);
     468             GMaterial* gmat = new GMaterial(name, index);
     469             gmat->setStandardDomain(standard_domain);
     470             addProperties(gmat, mat );
     471             gg->add(gmat);

     300 void AssimpGGeo::addProperties(GPropertyMap<float>* pmap, aiMaterial* material )
     301 {
     302     //unsigned int numProperties = material->mNumProperties ;
     303     for(unsigned int i = 0; i < material->mNumProperties; i++)
     304     {
     305         aiMaterialProperty* property = material->mProperties[i] ;
     306         aiString key = property->mKey ;
     307         const char* k = key.C_Str();
     308 
     309         // skip Assimp standard material props $clr.emissive $mat.shininess ?mat.name  etc..
     310         if( k[0] == '?' || k[0] == '$') continue ;
     311 
     312         //printf("AssimpGGeo::addProperties i %d k %s \n", i, k ); 
     313 
     314         aiPropertyTypeInfo type = property->mType ;
     315         if(type == aiPTI_Float)
     316         {
     317             addPropertyVector(pmap, k, property );
     318         }
     319         else if( type == aiPTI_String )
     320         {

     173 void AssimpGGeo::addPropertyVector(GPropertyMap<float>* pmap, const char* k, aiMaterialProperty* property )
     174 {
     175     const char* shortname = pmap->getShortName();
     176 
     177     LOG(debug) << "AssimpGGeo::addPropertyVector "
     178               << " shortname " << (shortname ? shortname : "-" )
     179               << " k " << k
     180                ;
     181 
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
     231 
     232     if(m_reverse)
     233     {
     234        std::reverse(vals.begin(), vals.end());
     235        std::reverse(domain.begin(), domain.end());
     236     }
     237 
     238     pmap->addProperty(k, vals.data(), domain.data(), vals.size() );
     239 }
     240 

::

    369 template <typename T>
    370 void GPropertyMap<T>::addProperty(const char* pname, T* values, T* domain, unsigned int length, const char* prefix)
    371 {
    372    //printf("GPropertyMap<T>::addProperty name %s pname %s length %u \n", getName(), pname, length );
    373    assert(length < 1000);
    374 
    375    GAry<T>* vals = new GAry<T>( length, values );
    376    GAry<T>* doms  = new GAry<T>( length, domain );
    377    GProperty<T>* orig = new GProperty<T>(vals, doms)  ;
    378 
    379    addPropertyStandardized(pname, orig, prefix);
    380 }
    381 

Interpolation onto standard domain happens right back at AssimpGGeo conversion from assimp ai props into GGeo::

    383 template <typename T>
    384 void GPropertyMap<T>::addPropertyStandardized(const char* pname,  GProperty<T>* orig, const char* prefix)
    385 {
    386    if(m_standard_domain)
    387    {
    388        GProperty<T>* ipol = orig->createInterpolatedProperty(m_standard_domain);
    389 
    390        //orig->Summary("orig", 10 );
    391        //ipol->Summary("ipol", 10 );
    392 
    393        addProperty(pname, ipol, prefix) ;
    394    }
    395    else
    396    {
    397        addProperty(pname, orig, prefix);
    398    }
    399 }
    400 


::

    simon:ggeo blyth$ opticks-find addPropertyStandardized
    ./ggeo/GMaterialLib.cc:        raw->addPropertyStandardized( GMaterialLib::refractive_index_local, rif ); 
    ./ggeo/GPropertyMap.cc:   addPropertyStandardized(pname, orig, prefix);
    ./ggeo/GPropertyMap.cc:void GPropertyMap<T>::addPropertyStandardized(const char* pname,  GProperty<T>* orig, const char* prefix)
    ./ggeo/tests/GMaterialLibTest.cc:        raw->addPropertyStandardized( GMaterialLib::refractive_index_local, f2 ); 
    ./ggeo/tests/GPropertyMapTest.cc:    pmap->addPropertyStandardized(ri, f2 );
    ./ggeo/GPropertyMap.hh:      void addPropertyStandardized(const char* pname,  GProperty<T>* orig, const char* prefix=NULL);
    simon:opticks blyth$ 










Possible Approaches to reduce interpolation mismatch
---------------------------------------------------------

* use more wavelength samples just for GROUPVEL, in separate groupvel texture,
  it is the only one that feeds directy into distributions without random shielding

* tighten wavelength pitch for all properties, current 20nm 

* move Opticks to energy domain interpolation, like Geant4,
  instead of current wavelength domain interpol  


Dumping Interpol deviations
-------------------------------


After moving to fine domain pitch (1nm) all the below deviations go to zero::

    np.all(rel == 0.)


The below are deviations obtained from interpolations at every 1nm 
using input raster of 20nm. 

::

    In [6]: run bnd.py
    [2016-11-18 14:38:53,114] p41098 {/Users/blyth/opticks/ana/base.py:210} INFO - envvar OPTICKS_ANA_DEFAULTS -> defaults {'src': 'torch', 'tag': '1', 'det': 'concentric'} 
    bnd.py
    [2016-11-18 14:38:53,123] p41098 {/Users/blyth/opticks/ana/proplib.py:149} WARNING - reshaped /tmp/blyth/opticks/InterpolationTest/OInterpolationTest_identity.npy from  (39, 984, 4) -> (123, 4, 2, 39, 4)  
    [2016-11-18 14:38:53,141] p41098 {/Users/blyth/opticks/ana/proplib.py:149} WARNING - reshaped /tmp/blyth/opticks/InterpolationTest/OInterpolationTest_interpol.npy from  (761, 984, 4) -> (123, 4, 2, 761, 4)  
    [2016-11-18 14:38:53,150] p41098 {/Users/blyth/opticks/ana/proplib.py:123} WARNING - direct names override
    [2016-11-18 14:38:53,151] p41098 {/Users/blyth/opticks/ana/proplib.py:139} WARNING - direct data override
    [2016-11-18 14:38:53,153] p41098 {/Users/blyth/opticks/ana/proplib.py:123} WARNING - direct names override
    [2016-11-18 14:38:53,153] p41098 {/Users/blyth/opticks/ana/proplib.py:139} WARNING - direct data override
    [2016-11-18 14:38:53,156] p41098 {/Users/blyth/opticks/ana/proplib.py:123} WARNING - direct names override
    [2016-11-18 14:38:53,156] p41098 {/Users/blyth/opticks/ana/proplib.py:139} WARNING - direct data override
    [2016-11-18 14:38:53,159] p41098 {/Users/blyth/opticks/ana/proplib.py:123} WARNING - direct names override
    [2016-11-18 14:38:53,159] p41098 {/Users/blyth/opticks/ana/proplib.py:139} WARNING - direct data override
    [2016-11-18 14:38:53,163] p41098 {/Users/blyth/opticks/ana/proplib.py:123} WARNING - direct names override
    [2016-11-18 14:38:53,164] p41098 {/Users/blyth/opticks/ana/proplib.py:139} WARNING - direct data override
    bnd.py:142: RuntimeWarning: invalid value encountered in divide
      rel = np.where( np.logical_or(avg < 1e-6, dif == 0), 0, dif/avg )
                                                       RINDEX                   ABSLEN                 RAYLEIGH                 REEMPROB                 GROUPVEL  
     0                      GdDopedLS      -0.0048     0.0053       -0.0096     0.0821        0.0000     0.0237       -0.0423     0.0032       -0.0125     0.0065  
     1             LiquidScintillator      -0.0048     0.0053       -0.0100     0.0821        0.0000     0.0237       -0.0423     0.0032       -0.0125     0.0065  
     2                        Acrylic      -0.0046     0.0053        0.0000     0.0968        0.0000     0.0237        0.0000     0.0000       -0.0123     0.0064  
     3                     MineralOil      -0.0046     0.0053       -0.0083     0.0232        0.0000     0.0237        0.0000     0.0000       -0.0123     0.0063  
     4                       Bialkali       0.0000     0.0000       -0.0396     0.0017        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
     5                       IwsWater      -0.0001     0.0000       -0.0084     0.0254        0.0000     0.0000        0.0000     0.0000       -0.0006     0.0005  
     6                          Water      -0.0001     0.0000       -0.0084     0.0254        0.0000     0.0000        0.0000     0.0000       -0.0006     0.0005  
     7                      DeadWater      -0.0001     0.0000       -0.0084     0.0254        0.0000     0.0000        0.0000     0.0000       -0.0006     0.0005  
     8                       OwsWater      -0.0001     0.0000       -0.0084     0.0254        0.0000     0.0000        0.0000     0.0000       -0.0006     0.0005  
     9                            ESR       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
    10                   OpaqueVacuum       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
    11                           Rock       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
    12                         Vacuum       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
    13                          Pyrex       0.0000     0.0000       -0.0396     0.0017        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
    14                            Air       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
    15                            PPE       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
    16                      Aluminium       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
    17          ADTableStainlessSteel       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
    18                           Foam       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
    19                       Nitrogen       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
    20                    NitrogenGas      -0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000       -0.0000     0.0000  
    21                          Nylon       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
    22                            PVC       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
    23                          Tyvek       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
    24                       Bakelite       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
    25                         MixGas       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
    26                           Iron       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
    27                         Teflon      -0.0046     0.0053        0.0000     0.0968        0.0000     0.0237        0.0000     0.0000       -0.0123     0.0064  
    28             UnstStainlessSteel       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
    29                            BPE       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
    30                          Ge_68       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
    31                          Co_60       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
    32                           C_13       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
    33                         Silver       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
    34                        RadRock       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  
    35                 StainlessSteel       0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000        0.0000     0.0000  


ABSLEN deviates for GdLS::

    In [43]: rel[0,0,:760,1].reshape(-1,20)
    Out[43]: 
    array([[ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.0004,  0.0007,  0.0013,  0.0015,  0.0017,  0.0018,  0.0018,  0.0022,  0.0022,  0.0021,  0.002 ,  0.0019,  0.002 ,  0.0018,  0.0015,  0.0012,  0.0009,  0.0008,  0.0004],
           [ 0.    ,  0.057 ,  0.0604,  0.0821,  0.0735,  0.0664,  0.0599,  0.054 ,  0.0577,  0.0512,  0.0451,  0.0392,  0.0335,  0.0339,  0.0281,  0.0225,  0.017 ,  0.0117,  0.0107,  0.0053],
           [ 0.    ,  0.0033,  0.0058,  0.0111,  0.0123,  0.0131,  0.0134,  0.0134,  0.0159,  0.0152,  0.0142,  0.0131,  0.0118,  0.0126,  0.0109,  0.009 ,  0.007 ,  0.0049,  0.0048,  0.0024],
           [ 0.    ,  0.0015,  0.0028,  0.0056,  0.0063,  0.0067,  0.007 ,  0.0071,  0.0086,  0.0083,  0.0079,  0.0074,  0.0067,  0.0073,  0.0064,  0.0053,  0.0042,  0.0029,  0.0029,  0.0015],
           [ 0.    ,  0.0009,  0.0017,  0.0036,  0.004 ,  0.0043,  0.0045,  0.0046,  0.0057,  0.0055,  0.0053,  0.0049,  0.0044,  0.005 ,  0.0043,  0.0036,  0.0028,  0.002 ,  0.0021,  0.0011],
           [ 0.    ,  0.0006,  0.0012,  0.0025,  0.0029,  0.0031,  0.0032,  0.0033,  0.0041,  0.004 ,  0.0038,  0.0035,  0.0032,  0.0037,  0.0032,  0.0027,  0.0021,  0.0014,  0.0015,  0.0008],
           [ 0.    ,  0.0005,  0.0009,  0.0019,  0.0022,  0.0023,  0.0024,  0.0025,  0.0032,  0.0031,  0.0029,  0.0027,  0.0025,  0.0028,  0.0025,  0.0021,  0.0016,  0.0011,  0.0012,  0.0006],
           [ 0.    ,  0.0004,  0.0007,  0.0015,  0.0017,  0.0018,  0.0019,  0.0019,  0.0025,  0.0024,  0.0023,  0.0022,  0.0019,  0.0023,  0.002 ,  0.0017,  0.0013,  0.0009,  0.001 ,  0.0005],
           [ 0.    ,  0.0054,  0.0088,  0.0191,  0.0193,  0.019 ,  0.0182,  0.0171,  0.0212,  0.0193,  0.0173,  0.0152,  0.0131,  0.0149,  0.0124,  0.0099,  0.0073,  0.0048,  0.0054,  0.0027],
           [ 0.    ,  0.002 ,  0.0035,  0.0085,  0.009 ,  0.0092,  0.0092,  0.0089,  0.0115,  0.0108,  0.0099,  0.0089,  0.0077,  0.0091,  0.0077,  0.0062,  0.0046,  0.003 ,  0.0036,  0.0018],
           [ 0.    ,  0.0153,  0.0193,  0.0386,  0.0345,  0.0309,  0.0276,  0.0245,  0.0299,  0.0262,  0.0226,  0.0193,  0.016 ,  0.0184,  0.015 ,  0.0116,  0.0084,  0.0053,  0.0063,  0.0031],
           [ 0.    ,  0.0277,  0.0268,  0.0499,  0.0416,  0.0356,  0.0307,  0.0266,  0.0324,  0.0279,  0.0238,  0.02  ,  0.0165,  0.019 ,  0.0154,  0.0119,  0.0085,  0.0053,  0.0065,  0.0032],
           [ 0.    ,  0.0016,  0.0028,  0.0076,  0.0079,  0.008 ,  0.0079,  0.0076,  0.0102,  0.0095,  0.0086,  0.0076,  0.0066,  0.008 ,  0.0067,  0.0054,  0.004 ,  0.0025,  0.0032,  0.0016],
           [ 0.    , -0.0001, -0.0001, -0.0004, -0.0004, -0.0004, -0.0004, -0.0004, -0.0006, -0.0006, -0.0006, -0.0005, -0.0005, -0.0006, -0.0005, -0.0004, -0.0003, -0.0002, -0.0003, -0.0002],
           [ 0.    ,  0.    ,  0.0001,  0.0002,  0.0002,  0.0002,  0.0002,  0.0002,  0.0003,  0.0003,  0.0003,  0.0003,  0.0002,  0.0003,  0.0003,  0.0002,  0.0002,  0.0001,  0.0002,  0.0001],
           [ 0.    ,  0.0003,  0.0005,  0.0016,  0.0017,  0.0017,  0.0017,  0.0017,  0.0025,  0.0023,  0.0021,  0.0019,  0.0017,  0.0022,  0.0019,  0.0015,  0.0011,  0.0007,  0.001 ,  0.0005],
           [ 0.    , -0.0003, -0.0005, -0.0016, -0.0017, -0.0018, -0.0018, -0.0018, -0.0028, -0.0027, -0.0025, -0.0023, -0.0021, -0.0028, -0.0024, -0.002 , -0.0015, -0.001 , -0.0014, -0.0008],
           [ 0.    ,  0.0001,  0.0001,  0.0004,  0.0004,  0.0004,  0.0004,  0.0004,  0.0006,  0.0005,  0.0005,  0.0005,  0.0004,  0.0005,  0.0005,  0.0004,  0.0003,  0.0002,  0.0003,  0.0001],
           [ 0.    , -0.    , -0.    , -0.0001, -0.0001, -0.0001, -0.0001, -0.0001, -0.0002, -0.0002, -0.0002, -0.0001, -0.0001, -0.0002, -0.0002, -0.0001, -0.0001, -0.0001, -0.0001, -0.    ],
           [ 0.    , -0.0002, -0.0003, -0.0013, -0.0014, -0.0014, -0.0015, -0.0014, -0.0022, -0.0021, -0.002 , -0.0018, -0.0016, -0.0022, -0.0019, -0.0016, -0.0012, -0.0007, -0.0012, -0.0006],
           [ 0.    , -0.0003, -0.0005, -0.002 , -0.0021, -0.0022, -0.0022, -0.0022, -0.0035, -0.0034, -0.0032, -0.0029, -0.0026, -0.0037, -0.0032, -0.0026, -0.002 , -0.0012, -0.002 , -0.0011],
           [ 0.    ,  0.    ,  0.    ,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0002,  0.0002,  0.0001,  0.0001,  0.0001,  0.0002,  0.0001,  0.0001,  0.0001,  0.    ,  0.0001,  0.    ],
           [ 0.    , -0.0005, -0.0009, -0.004 , -0.0043, -0.0046, -0.0047, -0.0047, -0.0079, -0.0078, -0.0075, -0.007 , -0.0063, -0.0096, -0.0086, -0.0073, -0.0056, -0.0034, -0.0066, -0.0037],
           [ 0.    ,  0.0002,  0.0004,  0.0019,  0.0019,  0.0019,  0.0018,  0.0017,  0.0028,  0.0026,  0.0023,  0.002 ,  0.0017,  0.0025,  0.0021,  0.0016,  0.0011,  0.0006,  0.0012,  0.0006],
           [ 0.    , -0.    , -0.0001, -0.0003, -0.0003, -0.0003, -0.0003, -0.0003, -0.0005, -0.0004, -0.0004, -0.0003, -0.0003, -0.0004, -0.0004, -0.0003, -0.0002, -0.0001, -0.0002, -0.0001],
           [ 0.    , -0.0002, -0.0004, -0.0019, -0.002 , -0.002 , -0.002 , -0.0019, -0.0033, -0.0031, -0.0029, -0.0026, -0.0023, -0.0035, -0.003 , -0.0024, -0.0017, -0.001 , -0.002 , -0.0011],
           [ 0.    , -0.0003, -0.0005, -0.0025, -0.0026, -0.0027, -0.0026, -0.0026, -0.0045, -0.0043, -0.004 , -0.0036, -0.0031, -0.0049, -0.0043, -0.0035, -0.0025, -0.0014, -0.003 , -0.0016],
           [ 0.    , -0.0001, -0.0003, -0.0014, -0.0015, -0.0015, -0.0015, -0.0014, -0.0025, -0.0023, -0.0021, -0.0019, -0.0016, -0.0026, -0.0022, -0.0017, -0.0012, -0.0007, -0.0014, -0.0008],
           [ 0.    , -0.0002, -0.0004, -0.0025, -0.0027, -0.0027, -0.0027, -0.0026, -0.0047, -0.0044, -0.0041, -0.0037, -0.0032, -0.0052, -0.0045, -0.0036, -0.0026, -0.0014, -0.0032, -0.0017],
           [ 0.    ,  0.0012,  0.0018,  0.01  ,  0.0092,  0.0083,  0.0073,  0.0063,  0.0105,  0.0091,  0.0077,  0.0063,  0.0049,  0.0075,  0.006 ,  0.0044,  0.0029,  0.0014,  0.0031,  0.0016],
           [ 0.    , -0.0001, -0.0001, -0.0007, -0.0007, -0.0007, -0.0007, -0.0006, -0.0011, -0.0011, -0.001 , -0.0008, -0.0007, -0.0011, -0.001 , -0.0007, -0.0005, -0.0003, -0.0006, -0.0003],
           [ 0.    , -0.0002, -0.0003, -0.002 , -0.0021, -0.0021, -0.002 , -0.0019, -0.0036, -0.0034, -0.0031, -0.0027, -0.0023, -0.0039, -0.0033, -0.0026, -0.0018, -0.0009, -0.0023, -0.0012]], dtype=float32)



GROUPVEL deviates for GdLS::

    In [44]: rel[0,1,:760,0].reshape(-1,20)
    Out[44]: 
    array([[ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    , -0.0006, -0.0011, -0.0018, -0.0022, -0.0025, -0.0027, -0.0028, -0.0032, -0.0032, -0.0032, -0.0031, -0.0029, -0.003 , -0.0027, -0.0023, -0.0019, -0.0015, -0.0012, -0.0006],
           [ 0.    , -0.0007, -0.0013, -0.0022, -0.0027, -0.003 , -0.0033, -0.0035, -0.004 , -0.004 , -0.004 , -0.0038, -0.0036, -0.0038, -0.0034, -0.0029, -0.0024, -0.0018, -0.0016, -0.0008],
           [ 0.    , -0.0019, -0.0036, -0.0064, -0.0077, -0.0089, -0.0098, -0.0104, -0.0123, -0.0125, -0.0125, -0.0122, -0.0117, -0.0124, -0.0114, -0.01  , -0.0083, -0.0063, -0.0057, -0.003 ],
           [ 0.    ,  0.0009,  0.0017,  0.0029,  0.0034,  0.0038,  0.0041,  0.0043,  0.0049,  0.0049,  0.0048,  0.0045,  0.0042,  0.0044,  0.0039,  0.0033,  0.0027,  0.002 ,  0.0018,  0.0009],
           [ 0.    ,  0.0012,  0.0022,  0.004 ,  0.0046,  0.0051,  0.0054,  0.0055,  0.0065,  0.0064,  0.0062,  0.0058,  0.0054,  0.0057,  0.005 ,  0.0043,  0.0034,  0.0025,  0.0023,  0.0012],
           [ 0.    , -0.0002, -0.0005, -0.0009, -0.001 , -0.0011, -0.0012, -0.0013, -0.0016, -0.0016, -0.0015, -0.0014, -0.0014, -0.0015, -0.0013, -0.0011, -0.0009, -0.0007, -0.0006, -0.0003],
           [ 0.    ,  0.    , -0.    ,  0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    ,  0.    , -0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    , -0.    , -0.    ,  0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    ,  0.    , -0.    , -0.    , -0.    ,  0.    ],
           [ 0.    ,  0.0002,  0.0003,  0.0007,  0.0008,  0.0008,  0.0009,  0.0009,  0.0012,  0.0011,  0.0011,  0.001 ,  0.0009,  0.0011,  0.0009,  0.0008,  0.0006,  0.0004,  0.0005,  0.0002],
           [ 0.    ,  0.0001,  0.0002,  0.0005,  0.0006,  0.0007,  0.0007,  0.0007,  0.0009,  0.0009,  0.0008,  0.0008,  0.0007,  0.0008,  0.0007,  0.0006,  0.0005,  0.0003,  0.0004,  0.0002],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    , -0.    ,  0.    ,  0.    ,  0.    , -0.    ,  0.    ,  0.    , -0.    ,  0.    ,  0.    , -0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    , -0.    , -0.    , -0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    , -0.    , -0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    , -0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.0001,  0.0001,  0.0001,  0.0002,  0.0002,  0.0002,  0.0002,  0.0002,  0.0002,  0.0002,  0.0002,  0.0002,  0.0002,  0.0002,  0.0001,  0.0001,  0.0001,  0.0001],
           [ 0.    ,  0.    ,  0.0001,  0.0002,  0.0002,  0.0002,  0.0002,  0.0002,  0.0003,  0.0003,  0.0002,  0.0002,  0.0002,  0.0003,  0.0002,  0.0002,  0.0001,  0.0001,  0.0001,  0.0001],
           [ 0.    ,  0.    ,  0.    ,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    ],
           [ 0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    ],
           [ 0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    , -0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.0001,  0.0001,  0.0001,  0.0001,  0.    ,  0.0001,  0.0001,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.0001,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    , -0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    , -0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    , -0.    , -0.    , -0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ],
           [ 0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ,  0.    ]], dtype=float32)




