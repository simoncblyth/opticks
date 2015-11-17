#include "GSurfaceLib.hh"

#include "GCache.hh"
#include "GOpticalSurface.hh"
#include "GSkinSurface.hh"
#include "GBorderSurface.hh"

#include "GItemList.hh"
#include "NPY.hpp"

#include <iomanip>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


// surface
const char* GSurfaceLib::detect            = "detect" ;
const char* GSurfaceLib::absorb            = "absorb" ;
const char* GSurfaceLib::reflect_specular  = "reflect_specular" ;
const char* GSurfaceLib::reflect_diffuse   = "reflect_diffuse" ;

const char* GSurfaceLib::REFLECTIVITY = "REFLECTIVITY" ;
const char* GSurfaceLib::EFFICIENCY   = "EFFICIENCY" ;
const char* GSurfaceLib::SENSOR_SURFACE = "SensorSurface" ;


const char* GSurfaceLib::keyspec = 
"detect:EFFICIENCY," 
"absorb:DUMMY," 
"reflect_specular:REFLECTIVITY," 
"reflect_diffuse:REFLECTIVITY," 
;

float GSurfaceLib::SURFACE_UNSET = -1.f ; 

void GSurfaceLib::save()
{
    saveToCache();
    saveOpticalBuffer();
}
GSurfaceLib* GSurfaceLib::load(GCache* cache)
{
    GSurfaceLib* lib = new GSurfaceLib(cache);
    lib->loadFromCache();
    lib->loadOpticalBuffer();
    return lib ; 
}

void GSurfaceLib::loadOpticalBuffer()
{
    std::string dir = getCacheDir(); 
    std::string name = getBufferName("Optical");
    NPY<unsigned int>* ibuf = NPY<unsigned int>::load(dir.c_str(), name.c_str()); 

    importOpticalBuffer(ibuf);
    setOpticalBuffer(ibuf); 
}
void GSurfaceLib::saveOpticalBuffer()
{
    NPY<unsigned int>* ibuf = createOpticalBuffer();
    saveToCache(ibuf, "Optical") ; 
    setOpticalBuffer(ibuf);
}




const char* GSurfaceLib::propertyName(unsigned int k)
{
    assert(k < 4);
    if(k == 0) return detect ;
    if(k == 1) return absorb ;
    if(k == 2) return reflect_specular;
    if(k == 3) return reflect_diffuse ;
    return "?" ;
}

void GSurfaceLib::Summary(const char* msg)
{
    LOG(info) << msg  
              << " NumSurfaces " << getNumSurfaces() 
              ;
}

void GSurfaceLib::defineDefaults(GPropertyMap<float>* defaults)
{
    defaults->addConstantProperty( detect          ,      SURFACE_UNSET );
    defaults->addConstantProperty( absorb          ,      SURFACE_UNSET );
    defaults->addConstantProperty( reflect_specular,      SURFACE_UNSET );
    defaults->addConstantProperty( reflect_diffuse ,      SURFACE_UNSET );
}

void GSurfaceLib::init()
{
    setKeyMap(keyspec);
    defineDefaults(getDefaults());
}


void GSurfaceLib::add(GBorderSurface* raw)
{
    GPropertyMap<float>* surf = dynamic_cast<GPropertyMap<float>* >(raw);
    add(surf);
}
void GSurfaceLib::add(GSkinSurface* raw)
{
    GPropertyMap<float>* surf = dynamic_cast<GPropertyMap<float>* >(raw);
    add(surf);
}

void GSurfaceLib::add(GPropertyMap<float>* surf)
{
    assert(!isClosed());
   
    GPropertyMap<float>* ssurf = createStandardSurface(surf) ;

    addDirect(ssurf);
}


void GSurfaceLib::addDirect(GPropertyMap<float>* surf)
{
    assert(!isClosed());
    m_surfaces.push_back(surf); 
}




bool GSurfaceLib::operator()(const GPropertyMap<float>* a_, const GPropertyMap<float>* b_)
{
    const char* a = a_->getShortName();
    const char* b = b_->getShortName();

    typedef std::map<std::string, unsigned int> MSU ;
    MSU& order = getOrder();   
    MSU::const_iterator end = order.end() ; 
 
    unsigned int ia = order.find(a) == end ? UINT_MAX :  order[a] ; 
    unsigned int ib = order.find(b) == end ? UINT_MAX :  order[b] ; 
    return ia < ib ; 
}

void GSurfaceLib::sort()
{
    typedef std::map<std::string, unsigned int> MSU ;
    MSU& order = getOrder();  

    if(order.size() == 0) return ; 
    std::stable_sort( m_surfaces.begin(), m_surfaces.end(), *this );
}


guint4 GSurfaceLib::createOpticalSurface(GPropertyMap<float>* src)
{
   assert(src->isSkinSurface() || src->isBorderSurface() || src->isSurface());
   GOpticalSurface* os = src->getOpticalSurface();
   assert(os && "all skin/boundary surface expected to have associated OpticalSurface");
   guint4 optical = os->getOptical();
   return optical ; 
}

guint4 GSurfaceLib::getOpticalSurface(unsigned int i)
{
    GPropertyMap<float>* surf = getSurface(i);    
    guint4 os = createOpticalSurface(surf);
    os.x = i ;
    return os ; 
}


GPropertyMap<float>* GSurfaceLib::createStandardSurface(GPropertyMap<float>* src)
{
    GProperty<float>* _detect           = NULL ; 
    GProperty<float>* _absorb           = NULL ; 
    GProperty<float>* _reflect_specular = NULL ; 
    GProperty<float>* _reflect_diffuse  = NULL ; 

    if(!src)
    {
        _detect           = getDefaultProperty(detect); 
        _absorb           = getDefaultProperty(absorb); 
        _reflect_specular = getDefaultProperty(reflect_specular); 
        _reflect_diffuse  = getDefaultProperty(reflect_diffuse); 
    }
    else
    {
        assert(getStandardDomain()->isEqual(src->getStandardDomain()));
        assert(src->isSurface());
        GOpticalSurface* os = src->getOpticalSurface() ;  // GSkinSurface and GBorderSurface ctor plant the OpticalSurface into the PropertyMap

        if(src->isSensor())
        {
            GProperty<float>* _EFFICIENCY = src->getProperty(EFFICIENCY); 
            assert(_EFFICIENCY && os && "sensor surfaces must have an efficiency" );

            if(m_fake_efficiency >= 0.f && m_fake_efficiency <= 1.0f)
            {
                _detect           = makeConstantProperty(m_fake_efficiency) ;    
                _absorb           = makeConstantProperty(1.0-m_fake_efficiency);
                _reflect_specular = makeConstantProperty(0.0);
                _reflect_diffuse  = makeConstantProperty(0.0);
            } 
            else
            {
                _detect = _EFFICIENCY ;      
                _absorb = GProperty<float>::make_one_minus( _detect );
                _reflect_specular = makeConstantProperty(0.0);
                _reflect_diffuse  = makeConstantProperty(0.0);
            }
        }
        else
        {
            GProperty<float>* _REFLECTIVITY = src->getProperty(REFLECTIVITY); 
            assert(_REFLECTIVITY && os && "non-sensor surfaces must have a reflectivity " );

            if(os->isSpecular())
            {
                _detect  = makeConstantProperty(0.0) ;    
                _reflect_specular = _REFLECTIVITY ;
                _reflect_diffuse  = makeConstantProperty(0.0) ;    
                _absorb  = GProperty<float>::make_one_minus(_reflect_specular);
            }
            else
            {
                _detect  = makeConstantProperty(0.0) ;    
                _reflect_specular = makeConstantProperty(0.0) ;    
                _reflect_diffuse  = _REFLECTIVITY ;
                _absorb  = GProperty<float>::make_one_minus(_reflect_diffuse);
            } 
        }
    }

    assert(_detect);
    assert(_absorb);
    assert(_reflect_specular);
    assert(_reflect_diffuse);

    GPropertyMap<float>* dst = new GPropertyMap<float>(src);

    dst->setSensor( src ? src->isSensor() : false ); 
    dst->addProperty( detect          , _detect          );
    dst->addProperty( absorb          , _absorb          );
    dst->addProperty( reflect_specular, _reflect_specular);
    dst->addProperty( reflect_diffuse , _reflect_diffuse );


    bool valid = checkSurface(dst);
    assert(valid);

    return dst ; 
}

GPropertyMap<float>* GSurfaceLib::makePerfect(const char* name, float detect_, float absorb_, float reflect_specular_, float reflect_diffuse_)
{
    GProperty<float>* _detect           = makeConstantProperty(detect_) ;    
    GProperty<float>* _absorb           = makeConstantProperty(absorb_) ;    
    GProperty<float>* _reflect_specular = makeConstantProperty(reflect_specular_) ;    
    GProperty<float>* _reflect_diffuse  = makeConstantProperty(reflect_diffuse_) ;    


    // placeholders
    const char* type = "1" ; 
    const char* model = "1" ; 
    const char* finish = "1" ; 
    const char* value = "1" ; 
    GOpticalSurface* os = new GOpticalSurface(name, type, model, finish, value);

    unsigned int index = 1000 ;   // does this matter ? 
    GPropertyMap<float>* dst = new GPropertyMap<float>(name, index, "surface", os);
    dst->setStandardDomain(getStandardDomain());

    dst->addProperty( detect          , _detect          );
    dst->addProperty( absorb          , _absorb          );
    dst->addProperty( reflect_specular, _reflect_specular);
    dst->addProperty( reflect_diffuse , _reflect_diffuse );
    return dst ;  
}


void GSurfaceLib::addPerfectSurfaces()
{
    GPropertyMap<float>* _detect = makePerfect("perfectDetectSurface", 1.f, 0.f, 0.f, 0.f );
    GPropertyMap<float>* _absorb = makePerfect("perfectAbsorbSurface", 0.f, 1.f, 0.f, 0.f );
    GPropertyMap<float>* _specular = makePerfect("perfectSpecularSurface", 0.f, 1.f, 0.f, 0.f );
    GPropertyMap<float>* _diffuse  = makePerfect("perfectDiffuseSurface", 0.f, 1.f, 0.f, 0.f );

    addDirect(_detect);
    addDirect(_absorb);
    addDirect(_specular);
    addDirect(_diffuse);
}




bool GSurfaceLib::checkSurface( GPropertyMap<float>* surf)
{
    // After standardization in GBoundaryLib surfaces must have four properties
    // that correspond to the probabilities of what happens at an intersection
    // with the surface.  These need to add to one across the wavelength domain.

    GProperty<float>* _detect = surf->getProperty(detect);
    GProperty<float>* _absorb = surf->getProperty(absorb);
    GProperty<float>* _reflect_specular = surf->getProperty(reflect_specular);
    GProperty<float>* _reflect_diffuse  = surf->getProperty(reflect_diffuse);

    if(!_detect) return false ; 
    if(!_absorb) return false ; 
    if(!_reflect_specular) return false ; 
    if(!_reflect_diffuse) return false ; 

    GProperty<float>* sum = GProperty<float>::make_addition( _detect, _absorb, _reflect_specular, _reflect_diffuse );
    GAry<float>* vals = sum->getValues();
    GAry<float>* ones = GAry<float>::ones(sum->getLength());
    float diff = GAry<float>::maxdiff( vals, ones );
    bool valid = diff < 1e-6 ; 

    if(!valid && surf->hasDefinedName())
    {
        LOG(warning) << "GSurfaceLib::checkSurface " << surf->getName() << " diff " << diff ; 
    }
    return valid ; 
}


GItemList* GSurfaceLib::createNames()
{
    GItemList* names = new GItemList(getType());
    unsigned int ni = getNumSurfaces();
    for(unsigned int i=0 ; i < ni ; i++)
    {
        GPropertyMap<float>* surf = m_surfaces[i] ;
        names->add(surf->getShortName());
    }
    return names ; 
}

NPY<float>* GSurfaceLib::createBuffer()
{
    unsigned int ni = getNumSurfaces();
    unsigned int nj = getStandardDomain()->getLength();
    unsigned int nk = 4 ; 
    assert(ni > 0 && nj > 0);

    NPY<float>* buf = NPY<float>::make(ni, nj, nk); 
    buf->zero();

    float* data = buf->getValues();

    GProperty<float> *p0,*p1,*p2,*p3 ; 

    for(unsigned int i=0 ; i < ni ; i++)
    {
        GPropertyMap<float>* surf = m_surfaces[i] ;
        p0 = surf->getPropertyByIndex(0);
        p1 = surf->getPropertyByIndex(1);
        p2 = surf->getPropertyByIndex(2);
        p3 = surf->getPropertyByIndex(3);

        for( unsigned int j = 0; j < nj; j++ ) // interleave 4 properties into the buffer
        {   
            unsigned int offset = i*nj*nk + j*nk ;  
            data[offset+0] = p0->getValue(j) ;
            data[offset+1] = p1->getValue(j) ;
            data[offset+2] = p2->getValue(j) ;
            data[offset+3] = p3->getValue(j) ;
        } 
    }
    return buf ; 
}



void GSurfaceLib::import()
{
    assert( m_buffer->getNumItems() == m_names->getNumKeys() );

    unsigned int ni = m_buffer->getShape(0);
    unsigned int nj = m_buffer->getShape(1);
    unsigned int nk = m_buffer->getShape(2);

    LOG(debug) << "GSurfaceLib::import "    
              << " ni " << ni 
              << " nj " << nj 
              << " nk " << nk
              ;

   assert(m_standard_domain->getLength() == nj );
   float* data = m_buffer->getValues();
   for(unsigned int i=0 ; i < ni ; i++)
   {
       const char* key = m_names->getKey(i);
       LOG(debug) << std::setw(3) << i 
                 << " " << key ;

       GOpticalSurface* os = NULL ;
       GPropertyMap<float>* surf = new GPropertyMap<float>(key,i,"surface", os);
       import(surf, data + i*nj*nk, nj, nk );

       m_surfaces.push_back(surf);
   }  
}

void GSurfaceLib::import( GPropertyMap<float>* surf, float* data, unsigned int nj, unsigned int nk )
{
    float* domain = m_standard_domain->getValues();

    for(unsigned int k = 0 ; k < nk ; k++)
    {
        float* values = new float[nj] ; 
        for(unsigned int j = 0 ; j < nj ; j++) values[j] = data[j*nk+k]; 
        GProperty<float>* prop = new GProperty<float>( values, domain, nj );
        surf->addProperty(propertyName(k), prop);
    } 
}


void GSurfaceLib::importOpticalBuffer(NPY<unsigned int>* ibuf)
{
    // invoked by load after loadFromCache and import have run 

    std::vector<guint4> optical ; 
    importUint4Buffer(optical, ibuf);  

    // thence can revivify GOpticalSurface and associate them to the GPropertyMap<float>*
    
    unsigned int ni = optical.size();
    assert(ni == getNumSurfaces());
    assert(ni == m_names->getNumKeys() );

    for(unsigned int i=0 ; i < ni ; i++)
    {
       const char* key = m_names->getKey(i);
       GPropertyMap<float>* surf = getSurface(i);
       GOpticalSurface* os = GOpticalSurface::create( key, optical[i] );
       surf->setOpticalSurface(os);
    }
}

NPY<unsigned int>* GSurfaceLib::createOpticalBuffer()
{
    std::vector<guint4> optical ; 
    unsigned int ni = getNumSurfaces();
    for(unsigned int i=0 ; i < ni ; i++) optical.push_back(getOpticalSurface(i));
    return createUint4Buffer(optical);
}



void GSurfaceLib::dump(const char* msg)
{
    Summary(msg);

    unsigned int ni = getNumSurfaces() ; 

    LOG(info) << " (index,type,finish,value) " ;  

    for(unsigned int i=0 ; i < ni ; i++)
    {
        guint4 optical = getOpticalSurface(i);
        GPropertyMap<float>* surf = getSurface(i);

        LOG(info) << std::setw(30) << surf->getName() 
                  << optical.description() 
                  ;
    } 

    int arg = m_cache->getLastArgInt();
    if(arg > -1)
        dump(arg);
    else
        for(unsigned int i=0 ; i < ni ; i++) dump(i);
}

void GSurfaceLib::dump( unsigned int index )
{
    guint4 optical = getOpticalSurface(index);
    GPropertyMap<float>* surf = getSurface(index);
    std::string desc = optical.description() + surf->description() ; 
    dump(surf, desc.c_str());
}


void GSurfaceLib::dump( GPropertyMap<float>* surf, const char* msg)
{
    GProperty<float>* _detect = surf->getProperty(detect);
    GProperty<float>* _absorb = surf->getProperty(absorb);
    GProperty<float>* _reflect_specular = surf->getProperty(reflect_specular);
    GProperty<float>* _reflect_diffuse  = surf->getProperty(reflect_diffuse);

    assert(_detect);
    assert(_absorb);
    assert(_reflect_specular);
    assert(_reflect_diffuse);

    std::string table = GProperty<float>::make_table( 
                            _detect, "detect", 
                            _absorb, "absorb",  
                            _reflect_specular, "reflect_specular",
                            _reflect_diffuse , "reflect_diffuse", 
                            20 );
    
    LOG(info) << msg << " " 
              << surf->getName()  
              << "\n" << table 
              ; 
}


bool GSurfaceLib::isSensorSurface(unsigned int surface)
{
    // name suffix based, see AssimpGGeo::convertSensor
    const char* name = getName(surface); 
    if(!name) return false ; 

    int pos = strlen(name) - strlen(SENSOR_SURFACE) ;
    bool iss = pos > 0 && strncmp(name + pos, SENSOR_SURFACE, strlen(SENSOR_SURFACE)) == 0 ;

    if(iss)
    LOG(debug) << "GSurfaceLib::isSensorSurface"
              << " surface " << surface  
              << " name " << name 
              << " pos " << pos 
              << " iss " << iss 
              ;

    return iss ; 
}

