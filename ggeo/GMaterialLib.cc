
#include <limits>

#include "SAbbrev.hh"
#include "BStr.hh"
#include "NMeta.hpp"
#include "NPY.hpp"
#include "Opticks.hh"

#include "GDomain.hh"
#include "GProperty.hh"
#include "GPropertyMap.hh"
#include "GMaterialLib.hh"
#include "GMaterial.hh"
#include "GItemList.hh"
#include "GConstant.hh"

#include "PLOG.hh"


const plog::Severity GMaterialLib::LEVEL = debug ;

const GMaterialLib* GMaterialLib::INSTANCE = NULL ; 
const GMaterialLib* GMaterialLib::GetInstance(){ return INSTANCE ; }


const float GMaterialLib::MATERIAL_UNSET   = 0.0f  ;

const char* GMaterialLib::refractive_index  = "refractive_index" ;
const char* GMaterialLib::absorption_length = "absorption_length" ;
const char* GMaterialLib::scattering_length = "scattering_length" ;
const char* GMaterialLib::reemission_prob   = "reemission_prob" ;

const char* GMaterialLib::group_velocity   = "group_velocity" ;
const char* GMaterialLib::extra_y          = "extra_y" ;
const char* GMaterialLib::extra_z          = "extra_z" ;
const char* GMaterialLib::extra_w          = "extra_w" ;

const char* GMaterialLib::refractive_index_local  = "RINDEX" ;

const char* GMaterialLib::keyspec = 
"refractive_index:RINDEX,"
"absorption_length:ABSLENGTH,"
"scattering_length:RAYLEIGH,"
"reemission_prob:REEMISSIONPROB," 
"group_velocity:GROUPVEL,"
"extra_y:EXTRA_Y,"
"extra_z:EXTRA_Z,"
"extra_w:EXTRA_W,"
"detect:EFFICIENCY,"
;


/**
m_keys = size=8 {
    [0] = "detect"
    [1] = "absorb"
    [2] = "reflect_specular"
    [3] = "reflect_diffuse"
    [4] = "extra_x"
    [5] = "extra_y"
    [6] = "extra_z"
    [7] = "extra_w"
  }

**/


void GMaterialLib::save()
{
    LOG(LEVEL) << "[" ; 
    saveToCache();
    LOG(LEVEL) << "]" ; 
}

GMaterialLib* GMaterialLib::load(Opticks* ok)
{
    GMaterialLib* mlib = new GMaterialLib(ok);
    mlib->loadFromCache();
    mlib->postLoadFromCache();
    return mlib ; 
}


void GMaterialLib::postLoadFromCache()
{
    bool nore = m_ok->hasOpt("nore") ;
    bool noab = m_ok->hasOpt("noab") ;
    bool nosc = m_ok->hasOpt("nosc") ;

    bool xxre = m_ok->hasOpt("xxre") ;
    bool xxab = m_ok->hasOpt("xxab") ;
    bool xxsc = m_ok->hasOpt("xxsc") ;

    bool fxre = m_ok->hasOpt("fxre") ;
    bool fxab = m_ok->hasOpt("fxab") ;
    bool fxsc = m_ok->hasOpt("fxsc") ;

    //bool groupvel = !m_ok->hasOpt("nogroupvel") ;

    LOG(LEVEL)
        << " nore " << nore 
        << " noab " << noab 
        << " nosc " << nosc 
        << " xxre " << xxre 
        << " xxab " << xxab 
        << " xxsc " << xxsc 
        << " fxre " << fxre 
        << " fxab " << fxab 
        << " fxsc " << fxsc 
    //  << " groupvel " << groupvel 
        ; 

    if(nore || xxre || fxre)
    {    
        float reemission_prob = 0.f ; 
        if(nore) reemission_prob = 0.f ;
        if(xxre) reemission_prob = 0.5f ;  // not 1.0 in order to leave some AB, otherwise too unphysical 
        if(fxre) reemission_prob = m_ok->getFxRe();

        LOG(fatal) << "GMaterialLib::postLoadFromCache --nore/--xxre/--fxre option postCache modifying reemission_prob " ; 
        setMaterialPropertyValues("GdDopedLS",          "reemission_prob", reemission_prob );
        setMaterialPropertyValues("LiquidScintillator", "reemission_prob", reemission_prob );
    }

    if(noab || xxab || fxab)
    {
        float absorption_length = 0.f ; 
        if(noab) absorption_length = std::numeric_limits<float>::max() ;
        if(xxab) absorption_length = 100.f ;
        if(fxab) absorption_length = m_ok->getFxAb();

        LOG(fatal) << "GMaterialLib::postLoadFromCache --noab/--xxab/--fxab option postCache modifying absorption_length: " << absorption_length ; 
        setMaterialPropertyValues("*",  "absorption_length", absorption_length );
    }

    if(nosc || xxsc || fxsc)
    {
        float scattering_length = 0.f ; 
        if(nosc) scattering_length = std::numeric_limits<float>::max() ;
        if(xxsc) scattering_length = 100.f ;
        if(fxsc) scattering_length = m_ok->getFxSc();

        LOG(fatal) << "GMaterialLib::postLoadFromCache --nosc/--xxsc/--fxsc option postCache modifying scattering_length: " << scattering_length ; 
        setMaterialPropertyValues("*",  "scattering_length", scattering_length );
    }



/*
    TRYING TO MOVE THIS beforeClose

    if(groupvel)  // unlike the other material changes : this one is ON by default, so long at not swiched off with --nogroupvel 
    {
       bool debug = false ; 
       replaceGROUPVEL(debug);
    }
    
*/



    if(nore || noab || nosc || xxre || xxab || xxsc || fxre || fxsc || fxab )
    {
        // need to replace the loaded buffer with a new one with the changes for Opticks to see it 
        NPY<float>* mbuf = createBuffer();
        setBuffer(mbuf);
    }

}


unsigned GMaterialLib::getNumMaterials() const  
{
    return m_materials.size();
}
unsigned GMaterialLib::getNumRawMaterials() const 
{
    return m_materials_raw.size();
}





GMaterialLib::GMaterialLib(Opticks* ok, GMaterialLib* basis) 
    :
    GPropertyLib(ok, "GMaterialLib"),
    m_basis(basis),
    m_cathode(NULL),
    m_cathode_material_name(NULL),
    m_material_order(ORDER_BY_PREFERENCE)
{
    init();
}

GMaterialLib::GMaterialLib(GMaterialLib* src, GDomain<float>* domain, GMaterialLib* basis)  // hmm think basis never used with this ctor ?
    :
    GPropertyLib(src, domain),
    m_basis(basis),
    m_cathode(NULL),
    m_cathode_material_name(NULL),
    m_material_order(ORDER_BY_PREFERENCE)
{
    init();
    initInterpolatingCopy(src, domain);
}
 

void GMaterialLib::init()
{
    INSTANCE = this ; 
    setKeyMap(keyspec);
    defineDefaults(getDefaults());
}

void GMaterialLib::initInterpolatingCopy(GMaterialLib* src, GDomain<float>* domain)
{
    unsigned nmat = src->getNumMaterials();

    for(unsigned i=0 ; i < nmat ; i++)
    {
        GMaterial* smat = src->getMaterial(i);

        GMaterial* dmat = new GMaterial(smat, domain );   // interpolating "copy" ctor

        addDirect(dmat);
    }
}


void GMaterialLib::defineDefaults(GPropertyMap<float>* defaults)
{
    defaults->addConstantProperty( refractive_index,      1.f  );
    defaults->addConstantProperty( absorption_length,     1e6  );
    defaults->addConstantProperty( scattering_length,     1e6  );
    defaults->addConstantProperty( reemission_prob,       0.f  );

    if(NUM_FLOAT4 > 1)
    {
        defaults->addConstantProperty( group_velocity,   300.f  );
        defaults->addConstantProperty( extra_y,          MATERIAL_UNSET  );
        defaults->addConstantProperty( extra_z,          MATERIAL_UNSET  );
        defaults->addConstantProperty( extra_w,          MATERIAL_UNSET  );
    }
}


const char* GMaterialLib::propertyName(unsigned int k)
{
    assert(k < 4*NUM_FLOAT4);

    if(k == 0) return refractive_index ;
    if(k == 1) return absorption_length ;
    if(k == 2) return scattering_length ;
    if(k == 3) return reemission_prob ;

    if(NUM_FLOAT4 > 1)
    {
        if(k == 4) return group_velocity ;
        if(k == 5) return extra_y ;
        if(k == 6) return extra_z ;
        if(k == 7) return extra_w ;
    }

    return "?" ;
}

void GMaterialLib::Summary(const char* msg)
{
    LOG(info) << msg  
              << " NumMaterials " << getNumMaterials() 
              << " NumFloat4 " << NUM_FLOAT4
              ;
}


// invoked pre-cache by GGeo::add(GMaterial* material) AssimpGGeo::convertMaterials
void GMaterialLib::add(GMaterial* mat)
{
    if(mat->hasProperty("EFFICIENCY"))
    {
        LOG(LEVEL) << " MATERIAL WITH EFFICIENCY " ; 
        setCathode(mat) ; 
    }

    bool with_lowercase_efficiency = mat->hasProperty("efficiency") ; 
    assert( !with_lowercase_efficiency ); 

    assert(!isClosed());
    m_materials.push_back(createStandardMaterial(mat)); 
}
void GMaterialLib::addRaw(GMaterial* mat)
{
    m_materials_raw.push_back(mat);
}

void GMaterialLib::addDirect(GMaterial* mat)
{
    assert(!isClosed());
    m_materials.push_back(mat); 
}

GMaterial* GMaterialLib::createStandardMaterial(GMaterial* src)
{
    assert(src);  // materials must always be defined
    assert(src->isMaterial());
    assert(getStandardDomain()->isEqual(src->getStandardDomain()));

    GMaterial* dst  = new GMaterial(src);

    if(dst->hasStandardDomain())
    {
        assert(dst->getStandardDomain()->isEqual(src->getStandardDomain()));
    }
    else
    {
        dst->setStandardDomain(src->getStandardDomain());
    }

    dst->addProperty(refractive_index, getPropertyOrDefault( src, refractive_index ));
    dst->addProperty(absorption_length,getPropertyOrDefault( src, absorption_length ));
    dst->addProperty(scattering_length,getPropertyOrDefault( src, scattering_length ));
    dst->addProperty(reemission_prob  ,getPropertyOrDefault( src, reemission_prob ));

    if(NUM_FLOAT4 > 1)
    {
        dst->addProperty(group_velocity, getPropertyOrDefault( src, group_velocity));
        dst->addProperty(extra_y       , getPropertyOrDefault( src, extra_y));
        dst->addProperty(extra_z       , getPropertyOrDefault( src, extra_z));
        dst->addProperty(extra_w       , getPropertyOrDefault( src, extra_w));
    }

    return dst ; 
}


/**
GMaterialLib::operator()
-------------------------

Defines the sort order of the materials based upon 
the order map keyed on the material short names.

The reason to rearrange the order, is to get important
materials at low indices to allow highly compressed step
by step recording of millions of photons on GPU using
only 4 bits to record the material index. 

**/


bool GMaterialLib::order_by_preference(const GMaterial& a_, const GMaterial& b_)
{
    const char* a = a_.getShortName();
    const char* b = b_.getShortName();

    typedef std::map<std::string, unsigned> MSU ;
    MSU& order = getOrder();  

    MSU::const_iterator end = order.end() ; 
    unsigned int ia = order.find(a) == end ? UINT_MAX :  order[a] ; 
    unsigned int ib = order.find(b) == end ? UINT_MAX :  order[b] ; 
    return ia < ib ; 
}


bool GMaterialLib::order_by_srcidx(const GMaterial& a_, const GMaterial& b_)
{
    // large default so test material additions which have no srcidx stay at the end
    int ia = a_.getMetaKV<int>("srcidx", "1000");
    int ib = b_.getMetaKV<int>("srcidx", "1000");
    return ia < ib ; 
}




bool GMaterialLib::operator()(const GMaterial& a_, const GMaterial& b_)
{
    bool order = false ; 
    switch(m_material_order)
    {
        case ORDER_ASIS          :  assert(0)                           ; break ; 
        case ORDER_BY_SRCIDX     :  order = order_by_srcidx(a_, b_)     ; break ; 
        case ORDER_BY_PREFERENCE :  order = order_by_preference(a_, b_) ; break ; 
    }
    return order ; 
}

const char* GMaterialLib::ORDER_ASIS_ = "ORDER_ASIS" ; 
const char* GMaterialLib::ORDER_BY_SRCIDX_     = "ORDER_BY_SRCIDX" ; 
const char* GMaterialLib::ORDER_BY_PREFERENCE_ = "ORDER_BY_PREFERENCE" ; 

const char* GMaterialLib::getMaterialOrdering() const 
{
    const char* s = NULL ; 
    switch(m_material_order)
    {
        case ORDER_ASIS          :  s = ORDER_ASIS_          ; break ; 
        case ORDER_BY_SRCIDX     :  s = ORDER_BY_SRCIDX_     ; break ; 
        case ORDER_BY_PREFERENCE :  s = ORDER_BY_PREFERENCE_ ; break ; 
    }
    return s ; 
}

/**
GMaterialLib::sort
--------------------

This is invoked from the base when the proplib is closed.

**/
void GMaterialLib::sort()
{
    LOG(LEVEL) << getMaterialOrdering() ; 

    if( m_material_order == ORDER_ASIS )
    {
        return ; 
    }
    else if( m_material_order == ORDER_BY_PREFERENCE )
    {
        typedef std::map<std::string, unsigned> MSU ;
        MSU& order = getOrder();  
        if(order.size() == 0) return ; 
    } 

    std::stable_sort( m_materials.begin(), m_materials.end(), *this );
}

GItemList* GMaterialLib::createNames()
{
    GItemList* names = new GItemList(getType());
    unsigned int ni = getNumMaterials();
    for(unsigned int i=0 ; i < ni ; i++)
    {
        GMaterial* mat = m_materials[i] ;
        names->add(mat->getShortName());
    }
    return names ;
}


unsigned int GMaterialLib::getMaterialIndex(const GMaterial* qmaterial)
{
    unsigned int ni = getNumMaterials();
    for(unsigned int i=0 ; i < ni ; i++)
    {
        GMaterial* mat = m_materials[i] ;
        if(mat == qmaterial) return i  ;
    }     
    return UINT_MAX ;  
}



NMeta* GMaterialLib::createMeta()
{
    LOG(LEVEL) << "." ; 
    NMeta* libmeta = new NMeta ; 
    unsigned int ni = getNumMaterials();

    std::vector<std::string> names ; 
    for(unsigned int i=0 ; i < ni ; i++)
    {
        GMaterial* mat = m_materials[i] ;
        const char* name = mat->getShortName();
        names.push_back(name); 
    }

    SAbbrev abbrev(names); 
    assert( abbrev.abbrev.size() == names.size() ); 
    assert( abbrev.abbrev.size() == ni ); 

    NMeta* abbrevmeta = new NMeta ; 
    for(unsigned i=0 ; i < ni ; i++)
    {
        const std::string& nm = names[i] ; 
        const std::string& ab = abbrev.abbrev[i] ; 
        abbrevmeta->set<std::string>(nm.c_str(), ab ) ; 
    }

    libmeta->setObj("abbrev", abbrevmeta );  
    return libmeta ; 
}


NPY<float>* GMaterialLib::createBuffer()
{
    return createBufferForTex2d() ; 
}

/**
GMaterialLib::createBufferForTex2d
------------------------------------

1. buffer is created of shape like (38, 2, 39, 4) where 38 is the number of materials
2. GProperty<float>  values from the m_materials vector of GMaterial
   are copied into the buffer 

Memory layout of this buffer is constrained at one end by 
requirements of tex2d<float4> and at the other by matching the number of
materials making it only possible to change in the middle number of groups.

**/

NPY<float>* GMaterialLib::createBufferForTex2d()
{
    unsigned int ni = getNumMaterials();
    unsigned int nj = NUM_FLOAT4 ; 
    unsigned int nk = getStandardDomain()->getLength();
    unsigned int nl = 4 ;

   
    if(ni == 0 || nj == 0)
    {
        LOG(error) << "GMaterialLib::createBufferForTex2d"
                   << " NO MATERIALS ? "
                   << " ni " << ni 
                   << " nj " << nj 
                   ;

        return NULL ;  
    } 

    NPY<float>* mbuf = NPY<float>::make(ni, nj, nk, nl);  // materials/payload-category/wavelength-samples/4prop
    mbuf->zero();
    float* data = mbuf->getValues();

    for(unsigned int i=0 ; i < ni ; i++)
    {
        GMaterial* mat = m_materials[i] ;
        GProperty<float> *p0,*p1,*p2,*p3 ; 

        for(unsigned int j=0 ; j < nj ; j++)
        {
            p0 = mat->getPropertyByIndex(j*4+0);
            p1 = mat->getPropertyByIndex(j*4+1);
            p2 = mat->getPropertyByIndex(j*4+2);
            p3 = mat->getPropertyByIndex(j*4+3);

            for( unsigned int k = 0; k < nk; k++ )    // over wavelength-samples
            {  
                unsigned int offset = i*nj*nk*nl + j*nk*nl + k*nl ;  

                data[offset+0] = p0 ? p0->getValue(k) : MATERIAL_UNSET ;
                data[offset+1] = p1 ? p1->getValue(k) : MATERIAL_UNSET ;
                data[offset+2] = p2 ? p2->getValue(k) : MATERIAL_UNSET ;
                data[offset+3] = p3 ? p3->getValue(k) : MATERIAL_UNSET ;
            }
        }
    }
    return mbuf ; 
}




void GMaterialLib::import()
{
    if(m_buffer == NULL)
    {
        setValid(false);
        LOG(warning) << "GMaterialLib::import NULL buffer " ; 
        return ;  
    }

    if( m_buffer->getNumItems() != m_names->getNumKeys() )
    {
        LOG(fatal) << "GMaterialLib::import numItems != numKeys "
                   << " buffer numItems " << m_buffer->getNumItems() 
                   << " names numKeys " << m_names->getNumKeys() 
                   ;
    }

    assert( m_buffer->getNumItems() == m_names->getNumKeys() );

    LOG(debug) << "GMaterialLib::import"    
              << " buffer shape " << m_buffer->getShapeString()
              ;

    importForTex2d();
}


/**
GMaterialLib::importForTex2d
------------------------------

m_materials vector of GMaterial instances
is reconstituted from the buffer


   (ni, nj, nk, nl)
   (38,  2, 39, 4) 

**/

void GMaterialLib::importForTex2d()
{
    unsigned int ni = m_buffer->getShape(0);
    unsigned int nj = m_buffer->getShape(1);
    unsigned int nk = m_buffer->getShape(2);
    unsigned int nl = m_buffer->getShape(3);

    unsigned int domLen = m_standard_domain->getLength() ;

    bool expected = domLen == nk && nj == NUM_FLOAT4 && nl == 4 ; 

    if(!expected )
        LOG(fatal) << "GMaterialLib::importForTex2d"
                   << " UNEXPECTED BUFFER SHAPE " 
                   << m_buffer->getShapeString()
                   << " domLen " << domLen 
                   << " nk " << nk 
                   << " (recreate geocache by running with -G)" 
                   ;

    assert(expected);

    float* data = m_buffer->getValues();

    for(unsigned int i=0 ; i < ni ; i++)
    {
        const char* key = m_names->getKey(i);
        LOG(debug) << std::setw(3) << i 
                   << " " << key ;

        GMaterial* mat = new GMaterial(key, i);

        for(unsigned int j=0 ; j < nj ; j++)  // over the two groups
        {
            import(mat, data + i*nj*nk*nl + j*nk*nl, nk, nl , j);
        }
        m_materials.push_back(mat);
    } 
}



void GMaterialLib::import( GMaterial* mat, float* data, unsigned nj, unsigned nk, unsigned jcat )
{
    float* domain = m_standard_domain->getValues();

    for(unsigned k = 0 ; k < nk ; k++)  // over 4 props of the float4 
    {
        float* values = new float[nj] ; 
        for(unsigned j = 0 ; j < nj ; j++) values[j] = data[j*nk+k];   // un-interleaving 
       
        GProperty<float>* prop = new GProperty<float>( values, domain, nj );  
        mat->addProperty(propertyName(k+4*jcat), prop);
    } 
}


void GMaterialLib::beforeClose()
{
    LOG(LEVEL) << "." ; 
    bool debug = false ; 
    replaceGROUPVEL(debug); 
}



/**
GMaterialLib::replaceGROUPVEL
------------------------------

Use the refractive index of each Opticks GMaterial to 
calculate the groupvel and replace the GROUPVEL property of the material. 

Huh : requiring the buffer is odd... it prevents
invoking this before the close.

Fixed this to not require the buffer so it can be invoked before 
saving to cache or after. 

**/

void GMaterialLib::replaceGROUPVEL(bool debug)
{
    //unsigned ni = m_buffer->getShape(0);
    unsigned ni = getNumMaterials() ; // from the vector

    LOG(LEVEL) << "GMaterialLib::replaceGROUPVEL " << " ni " << ni ;

    const char* base = "$TMP/replaceGROUPVEL" ;

    if(debug)
    {
        LOG(warning) << "GMaterialLib::replaceGROUPVEL debug active : saving refractive_index.npy and group_velocity.npy beneath " << base  ; 
    }


    /*
    for(unsigned i=0 ; i < ni ; i++) 
        std::cout 
             << " i " << std::setw(2) << i 
             << m_materials[i]->dump_ptr()
             << std::endl 
             ;
    */


    for(unsigned i=0 ; i < ni ; i++)
    {
        GMaterial* mat = getMaterial(i);
        assert( m_materials[i] == mat ); 

        const char* key = mat->getName() ; 

        GProperty<float>* vg = mat->getProperty(group_velocity);
        GProperty<float>* ri = mat->getProperty(refractive_index);
        assert(vg);
        assert(ri);

        GProperty<float>* vg_calc = GProperty<float>::make_GROUPVEL(ri);
        assert(vg_calc);

        LOG(verbose)
            << " i " << std::setw(2) << i 
            << " n " << std::setw(2) << m_materials.size() 
            << " mat " << mat 
            << " vg " << vg
            << " ri " << ri
            << " vg_calc " << vg_calc
            << " k " << key
            ; 

        float wldelta = 1e-4 ; 
        vg->copyValuesFrom(vg_calc, wldelta);         // <-- replacing the property inside the GMaterial instance

        if(debug)
        {
            LOG(info) << " i " << std::setw(3) << i 
                      << " key " << std::setw(35) << key 
                      << " vg " << vg 
                      << " vg_calc " << vg_calc 
                      ;

            dump(mat, "replaceGROUPVEL");
            ri->save(base, key, "refractive_index.npy" );  
            vg_calc->save(base, key, "vg_calc.npy" );  
            vg->save(base, key, "group_velocity.npy" );  
        }

    }
}


bool GMaterialLib::setMaterialPropertyValues(const char* matname, const char* propname, float val)
{
    bool ret = true ; 

    if(matname && matname[0] == '*')
    {
         unsigned ni = m_buffer->getShape(0);
         LOG(info) << " wildcard GMaterialLib::setMaterialPropertyValues " << propname << " val " << val << " ni " << ni ; 
         for(unsigned i=0 ; i < ni ; i++)
         {
              const char* key = m_names->getKey(i);
              GMaterial* mat = getMaterial(key);
              LOG(info) << " i " << std::setw(3) << i 
                        << " key " << std::setw(35) << key 
                        ;

              ret = mat->setPropertyValues( propname, val );
         }
    }
    else
    {
        GMaterial* mat = getMaterial(matname);
        if(!mat) 
        {
             LOG(fatal) << " no material with name " << matname ; 
             assert(0);
             return false ; 
        } 
        ret = mat->setPropertyValues( propname, val );
        assert(ret);
    }

    return ret ; 
}








void GMaterialLib::dump(const char* msg)
{
    Summary(msg);

    unsigned int ni = getNumMaterials() ; 

    int index = m_ok->getLastArgInt();
    const char* lastarg = m_ok->getLastArg();

    if(hasMaterial(index))
    {
        dump(index);
    } 
    else if(hasMaterial(lastarg))
    {
        GMaterial* mat = getMaterial(lastarg);
        dump(mat);
    }
    else
    {
        for(unsigned int i=0 ; i < ni ; i++) dump(i);
    }
}


void GMaterialLib::dump(unsigned int index)
{
    GMaterial* mat = getMaterial(index);
    dump(mat);
}


// static 
void GMaterialLib::dump(GMaterial* mat)
{
    dump(mat, mat->description().c_str());
}

// static 
void GMaterialLib::dump( GMaterial* mat, const char* msg)
{
    GProperty<float>* _refractive_index = mat->getProperty(refractive_index);
    GProperty<float>* _absorption_length = mat->getProperty(absorption_length);
    GProperty<float>* _scattering_length = mat->getProperty(scattering_length);
    GProperty<float>* _reemission_prob = mat->getProperty(reemission_prob);
    GProperty<float>* _group_velocity = mat->getProperty(group_velocity);


    unsigned int fw = 20 ;
    float dscale = 1.0f ;
    bool dreciprocal = false  ; 

    //f loat(GConstant::h_Planck*GConstant::c_light/GConstant::nanometer) ;
    // no need for scaling or taking reciprocal, as GMaterial domains 
    // are already  in nm

    std::string table = GProperty<float>::make_table( 
                            fw,  dscale, dreciprocal,  
                            _refractive_index, refractive_index, 
                            _absorption_length, absorption_length,  
                            _scattering_length, scattering_length,  
                            _reemission_prob, reemission_prob, 
                            _group_velocity, group_velocity
                            );

    std::cout << table << std::endl ; 

    LOG(warning) << msg << " " << mat->getName()  
                 ;
}





const char* GMaterialLib::getNameCheck(unsigned int i)
{
    GMaterial* mat = getMaterial(i);
    const char* name1 =  mat->getShortName();
    const char* name2 = getName(i);
    assert(strcmp(name1, name2) == 0);

    return name1 ; 
}



bool GMaterialLib::hasMaterial(const char* name) const 
{
    return getMaterial(name) != NULL ; 
}

bool GMaterialLib::hasMaterial(unsigned int index) const 
{
    return getMaterial(index) != NULL ; 
}


/*
// this old way only works after closing geometry, or from loaded 
// as uses the names buffer

GMaterial* GMaterialLib::getMaterial(const char* name)
{
    unsigned int index = getIndex(name);
    return getMaterial(index);   
}
*/


//
// NB cannot get a live index whilst still adding 
// as materials are priority order sorted on GPropertyLib::close
//


GMaterial* GMaterialLib::getMaterial(const char* name) const 
{
    if(!name) return NULL ; 

    GMaterial* mat = NULL ; 
    for(unsigned i=0 ; i < m_materials.size() ; i++)
    {
        GMaterial* m = m_materials[i] ; 
        const char* mname = m->getName() ; 
        assert( mname ) ; 
        if( strcmp( mname, name ) == 0 )
        {
             mat = m ; 
             break ; 
        } 
    }
    return mat ; 
}

GMaterial* GMaterialLib::getMaterial(unsigned int index) const 
{
    return index < m_materials.size() ? m_materials[index] : NULL  ;
}

GMaterial* GMaterialLib::getMaterialWithIndex(unsigned aindex) const 
{
    GMaterial* mat = NULL ; 
    for(unsigned int i=0 ; i < m_materials.size() ; i++ )
    { 
        if(m_materials[i]->getIndex() == aindex )
        {
            mat = m_materials[i] ; 
            break ; 
        }
    }
    return mat ;
}


GPropertyMap<float>* GMaterialLib::findRawMaterial(const char* shortname) const 
{
    GMaterial* mat = NULL ; 
    for(unsigned int i=0 ; i < m_materials_raw.size() ; i++ )
    { 
        std::string sn = m_materials_raw[i]->getShortNameString();
        //printf("GGeo::findRawMaterial %d %s \n", i, sn.c_str()); 
        if(strcmp(sn.c_str(), shortname)==0)
        {
            mat = m_materials_raw[i] ; 
            break ; 
        }
    }
    return (GPropertyMap<float>*)mat ;
}

GProperty<float>* GMaterialLib::findRawMaterialProperty(const char* shortname, const char* propname) const
{
    GPropertyMap<float>* mat = findRawMaterial(shortname);

    GProperty<float>* prop = mat->getProperty(propname);

    prop->Summary();

    return prop ;   
}

void GMaterialLib::dumpRawMaterialProperties(const char* msg) const 
{
    LOG(info) << msg ; 
    for(unsigned int i=0 ; i < m_materials_raw.size() ; i++)
    {
        GMaterial* mat = m_materials_raw[i];
        //mat->Summary();
        std::cout << std::setw(30) << mat->getShortName()
                  << " keys: " << mat->getKeysString()
                  << std::endl ; 
    }
}

std::vector<GMaterial*> GMaterialLib::getRawMaterialsWithProperties(const char* props, char delim) const 
{
    std::vector<std::string> elem ;
    BStr::split(elem, props, delim);

    LOG(LEVEL)
         << props 
         << " m_materials_raw.size()  " << m_materials_raw.size() 
         ; 

    std::vector<GMaterial*>  selected ; 
    for(unsigned int i=0 ; i < m_materials_raw.size() ; i++)
    {
        GMaterial* mat = m_materials_raw[i];
        unsigned int found(0);
        for(unsigned int p=0 ; p < elem.size() ; p++)
        { 
           if(mat->hasProperty(elem[p].c_str())) found+=1 ;        
        }
        if(found == elem.size()) selected.push_back(mat);
    }
    return selected ;  
}








GMaterial* GMaterialLib::getBasisMaterial(const char* name) const 
{
    return m_basis ? m_basis->getMaterial(name) : NULL ; 
}

void GMaterialLib::reuseBasisMaterial(const char* name)
{
    GMaterial* mat = getBasisMaterial(name);
    if(!mat) LOG(fatal) << "reuseBasisMaterial requires basis library to be present and to contain the material  " << name ; 
    assert( mat );        

    addDirect( mat ); 
}




GMaterial* GMaterialLib::makeRaw(const char* name)
{
    GMaterial* raw = new GMaterial(name, getNumMaterials() );
    return raw ; 
}

void GMaterialLib::addTestMaterials()
{
    typedef std::pair<std::string, std::string> SS ; 
    typedef std::vector<SS> VSS ; 

    VSS rix ; 

    rix.push_back(SS("GlassSchottF2", "$OPTICKS_INSTALL_PREFIX/opticksdata/refractiveindex/tmp/glass/schott/F2.npy"));
    rix.push_back(SS("MainH2OHale",   "$OPTICKS_INSTALL_PREFIX/opticksdata/refractiveindex/tmp/main/H2O/Hale.npy"));
    
    // NB when adding test materials also need to set in prefs ~/.opticks/GMaterialLib
    //
    //    * priority order (for transparent materials arrange to be less than 16 for material sequence tracking)
    //    * color 
    //    * two letter abbreviation
    //
    // for these settings to be acted upon must rebuild the geocache with : "ggv -G"      
    //

    for(VSS::const_iterator it=rix.begin() ; it != rix.end() ; it++)
    {
        std::string name = it->first ; 
        std::string path = it->second ; 

        LOG(info) << "GMaterialLib::addTestMaterials" 
                  << " name " << std::setw(30) << name 
                  << " path " << path 
                  ;

        GProperty<float>* rif = GProperty<float>::load(path.c_str());
        if(!rif) continue ; 

        GMaterial* raw = makeRaw(name.c_str());
        raw->addPropertyStandardized( GMaterialLib::refractive_index_local, rif ); 
        
        add(raw);
   } 
}







void GMaterialLib::setCathode(GMaterial* cathode)
{
    assert( cathode ) ; 
    if( cathode && m_cathode && cathode == m_cathode )
    {
        LOG(fatal) << " have already set that cathode GMaterial : " << cathode->getName() ; 
        return ; 
    }
    if( cathode && m_cathode && cathode != m_cathode )
    {
        LOG(fatal) << " not expecting to change cathode GMaterial from  "
                   << m_cathode->getName()
                   << " to " 
                   << cathode->getName()
                   ; 
        assert(0); 
    } 
    LOG(LEVEL)
           << " setting cathode " 
           << " GMaterial : " << cathode 
           << " name : " << cathode->getName() ; 
    //cathode->Summary();       
    LOG(LEVEL) << cathode->prop_desc() ; 

    assert( cathode->hasNonZeroProperty("EFFICIENCY") );  

    m_cathode = cathode ; 
    m_cathode_material_name = strdup( cathode->getName() ) ; 
}
GMaterial* GMaterialLib::getCathode() const 
{
    return m_cathode ; 
}

const char* GMaterialLib::getCathodeMaterialName() const
{
    return m_cathode_material_name ; 
}






