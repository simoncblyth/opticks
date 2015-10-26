#include "GMaterialLib.hh"
#include "GMaterial.hh"
#include "GItemList.hh"
#include "NPY.hpp"

#include <iomanip>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


const char* GMaterialLib::refractive_index  = "refractive_index" ;
const char* GMaterialLib::absorption_length = "absorption_length" ;
const char* GMaterialLib::scattering_length = "scattering_length" ;
const char* GMaterialLib::reemission_prob   = "reemission_prob" ;

const char* GMaterialLib::keyspec = 
"refractive_index:RINDEX,"
"absorption_length:ABSLENGTH,"
"scattering_length:RAYLEIGH,"
"reemission_prob:REEMISSIONPROB," 
;


const char* GMaterialLib::propertyName(unsigned int i)
{
    assert(i < 4);
    if(i == 0) return refractive_index ;
    if(i == 1) return absorption_length ;
    if(i == 2) return scattering_length ;
    if(i == 3) return reemission_prob ;
    return "?" ;
}

void GMaterialLib::Summary(const char* msg)
{
    LOG(info) << msg  
              << " NumMaterials " << getNumMaterials() 
              << " NumRawMaterials " << getNumRawMaterials() 
              ;
}

void GMaterialLib::defineDefaults(GPropertyMap<float>* defaults)
{
    defaults->addConstantProperty( refractive_index,      1.f  );
    defaults->addConstantProperty( absorption_length,     1e6  );
    defaults->addConstantProperty( scattering_length,     1e6  );
    defaults->addConstantProperty( reemission_prob,       0.f  );
}

void GMaterialLib::init()
{
    setKeyMap(keyspec);
    defineDefaults(getDefaults());
}

void GMaterialLib::add(GMaterial* raw)
{
    m_materials_raw.push_back(raw);
    m_materials.push_back(createStandardMaterial(raw)); 
}

GMaterial* GMaterialLib::createStandardMaterial(GMaterial* src)
{
    assert(src);  // materials must always be defined
    assert(src->isMaterial());
    assert(getStandardDomain()->isEqual(src->getStandardDomain()));

    GMaterial* dst  = new GMaterial(src);

    if(dst->hasStandardDomain())
        assert(dst->getStandardDomain()->isEqual(src->getStandardDomain()));
    else
        dst->setStandardDomain(src->getStandardDomain());

    dst->addProperty(refractive_index, getPropertyOrDefault( src, refractive_index ));
    dst->addProperty(absorption_length,getPropertyOrDefault( src, absorption_length ));
    dst->addProperty(scattering_length,getPropertyOrDefault( src, scattering_length ));
    dst->addProperty(reemission_prob  ,getPropertyOrDefault( src, reemission_prob ));

    return dst ; 
}


void GMaterialLib::createBuffer()
{
    unsigned int ni = getNumMaterials();
    unsigned int nj = getStandardDomain()->getLength();
    unsigned int nk = 4 ; 
    assert(ni > 0 && nj > 0);

    GItemList* names = new GItemList(getType());
    NPY<float>* mbuf = NPY<float>::make(ni, nj, nk, NULL); 
    mbuf->zero();

    float* data = mbuf->getValues();

    GProperty<float> *p0,*p1,*p2,*p3 ; 

    for(unsigned int i=0 ; i < ni ; i++)
    {
        GMaterial* mat = m_materials[i] ;
        names->add(mat->getShortName());

        p0 = mat->getPropertyByIndex(0);
        p1 = mat->getPropertyByIndex(1);
        p2 = mat->getPropertyByIndex(2);
        p3 = mat->getPropertyByIndex(3);

        for( unsigned int j = 0; j < nj; j++ ) // interleave 4 properties into the buffer
        {   
            unsigned int offset = i*nj*nk + j*nk ;  

            data[offset+0] = p0->getValue(j) ;
            data[offset+1] = p1->getValue(j) ;
            data[offset+2] = p2->getValue(j) ;
            data[offset+3] = p3->getValue(j) ;
        } 
    }

    setBuffer(mbuf);
    setNames(names);
}

void GMaterialLib::import()
{
    assert( m_buffer->getNumItems() == m_names->getNumItems() );

    unsigned int ni = m_buffer->getShape(0);
    unsigned int nj = m_buffer->getShape(1);
    unsigned int nk = m_buffer->getShape(2);

    LOG(info) << " GMaterialLib::import "    
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

       GMaterial* mat = new GMaterial(name.c_str(), i);
       import(mat, data + i*nj*nk, nj, nk );

       m_materials.push_back(mat);
   }  
}

void GMaterialLib::import( GMaterial* mat, float* data, unsigned int nj, unsigned int nk )
{
    float* domain = m_standard_domain->getValues();

    for(unsigned int k = 0 ; k < nk ; k++)
    {
        float* values = new float[nj] ; 
        for(unsigned int j = 0 ; j < nj ; j++) values[j] = data[j*nk+k]; 
        GProperty<float>* prop = new GProperty<float>( values, domain, nj );
        mat->addProperty(propertyName(k), prop);
    } 
}


void GMaterialLib::dump(const char* msg)
{
    Summary(msg);

    unsigned int ni = getNumMaterials() ; 
    for(unsigned int i=0 ; i < ni ; i++)
    {
        GMaterial* mat = getMaterial(i);
        dump(mat, mat->description().c_str());
    }
}


void GMaterialLib::dump( GMaterial* mat, const char* msg)
{
    GProperty<float>* _refractive_index = mat->getProperty(refractive_index);
    GProperty<float>* _absorption_length = mat->getProperty(absorption_length);
    GProperty<float>* _scattering_length = mat->getProperty(scattering_length);
    GProperty<float>* _reemission_prob = mat->getProperty(reemission_prob);


    std::string table = GProperty<float>::make_table( 
                            _refractive_index, "refractive_index", 
                            _absorption_length, "absorption_length",  
                            _scattering_length, "scattering_length",  
                            _reemission_prob, "reemission_prob", 
                            20 );
    
    LOG(info) << msg << " " 
              << mat->getName()  
              << "\n" << table 
              ; 
}



GMaterialLib* GMaterialLib::load(GCache* cache)
{
    GMaterialLib* mlib = new GMaterialLib(cache);
    mlib->loadFromCache();
    return mlib ; 
}






