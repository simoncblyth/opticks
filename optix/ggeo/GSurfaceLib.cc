#include "GSurfaceLib.hh"

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


const char* GSurfaceLib::keyspec = 
"detect:EFFICIENCY," 
"absorb:DUMMY," 
"reflect_specular:REFLECTIVITY," 
"reflect_diffuse:REFLECTIVITY," 
;

float GSurfaceLib::SURFACE_UNSET = -1.f ; 



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
              << " NumRawSurfaces " << getNumRawSurfaces() 
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
    m_surfaces_raw.push_back(surf);
    m_surfaces.push_back(createStandardSurface(surf)); 
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
        assert(src->isSkinSurface() || src->isBorderSurface());
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


void GSurfaceLib::createBuffer()
{
    unsigned int ni = getNumSurfaces();
    unsigned int nj = getStandardDomain()->getLength();
    unsigned int nk = 4 ; 
    assert(ni > 0 && nj > 0);

    GItemList* names = new GItemList(getType());
    NPY<float>* buf = NPY<float>::make(ni, nj, nk, NULL); 
    buf->zero();

    float* data = buf->getValues();

    GProperty<float> *p0,*p1,*p2,*p3 ; 

    for(unsigned int i=0 ; i < ni ; i++)
    {
        GPropertyMap<float>* surf = m_surfaces[i] ;
        names->add(surf->getShortName());

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

    setBuffer(buf);
    setNames(names);
}

void GSurfaceLib::import()
{
    assert( m_buffer->getNumItems() == m_names->getNumItems() );

    unsigned int ni = m_buffer->getShape(0);
    unsigned int nj = m_buffer->getShape(1);
    unsigned int nk = m_buffer->getShape(2);

    LOG(info) << " GSurfaceLib::import "    
              << " ni " << ni 
              << " nj " << nj 
              << " nk " << nk
              ;

   assert(m_standard_domain->getLength() == nj );
   float* data = m_buffer->getValues();
   for(unsigned int i=0 ; i < ni ; i++)
   {
       std::string name = m_names->getItem(i);
       LOG(info) << std::setw(3) << i 
                 << " " << name ;

       GPropertyMap<float>* surf = new GPropertyMap<float>(name.c_str());
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


void GSurfaceLib::dump(const char* msg)
{
    Summary(msg);

    unsigned int ni = getNumSurfaces() ; 
    for(unsigned int i=0 ; i < ni ; i++)
    {
        GPropertyMap<float>* surf = getSurface(i);
        dump(surf, surf->description().c_str());
    }
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




GSurfaceLib* GSurfaceLib::load(GCache* cache)
{
    GSurfaceLib* lib = new GSurfaceLib(cache);
    lib->loadFromCache();
    return lib ; 
}


