#include "GMaterialLib.hh"
#include "GMaterial.hh"
#include "GItemList.hh"
#include "NPY.hpp"


#include <iomanip>
#include <boost/algorithm/string.hpp>
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


void GMaterialLib::save()
{
    saveToCache();
}

GMaterialLib* GMaterialLib::load(GCache* cache)
{
    GMaterialLib* mlib = new GMaterialLib(cache);
    mlib->loadFromCache();
    return mlib ; 
}

void GMaterialLib::init()
{
    setKeyMap(keyspec);
    defineDefaults(getDefaults());
}

void GMaterialLib::defineDefaults(GPropertyMap<float>* defaults)
{
    defaults->addConstantProperty( refractive_index,      1.f  );
    defaults->addConstantProperty( absorption_length,     1e6  );
    defaults->addConstantProperty( scattering_length,     1e6  );
    defaults->addConstantProperty( reemission_prob,       0.f  );
}


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
              ;
}
void GMaterialLib::add(GMaterial* raw)
{
    assert(!isClosed());
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


bool GMaterialLib::operator()(const GMaterial& a_, const GMaterial& b_)
{
    const char* a = a_.getShortName();
    const char* b = b_.getShortName();
    std::map<std::string, unsigned int>::const_iterator end = m_order.end() ; 
    unsigned int ia = m_order.find(a) == end ? UINT_MAX :  m_order[a] ; 
    unsigned int ib = m_order.find(b) == end ? UINT_MAX :  m_order[b] ; 
    return ia < ib ; 
}

void GMaterialLib::sort()
{
    if(m_order.size() == 0) return ; 
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

NPY<float>* GMaterialLib::createBuffer()
{
    unsigned int ni = getNumMaterials();
    unsigned int nj = getStandardDomain()->getLength();
    unsigned int nk = 4 ; 
    assert(ni > 0 && nj > 0);

    NPY<float>* mbuf = NPY<float>::make(ni, nj, nk); 
    mbuf->zero();

    float* data = mbuf->getValues();

    GProperty<float> *p0,*p1,*p2,*p3 ; 

    for(unsigned int i=0 ; i < ni ; i++)
    {
        GMaterial* mat = m_materials[i] ;

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
    return mbuf ; 
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


void GMaterialLib::dumpMaterials(const char* mats, const char* msg)
{
    typedef std::vector<std::string> VS ; 
    VS elem ; 
    boost::split(elem, mats, boost::is_any_of(","));

    LOG(info) << msg << " " << mats ; 
    for(VS::const_iterator it=elem.begin() ; it != elem.end() ; it++)
    {
        const char* key = it->c_str();
        unsigned int idx = getIndex(key);
        if(idx == GPropertyLib::UNSET)
        {
             LOG(warning) << "GMaterialLib::dumpMaterials no material named: " << *it ; 
        }
        else
        {
             const char* colorname = getColorName(key);  
             unsigned int colorcode = getColorCode(key);              

             std::cout << std::setw(5) << idx 
                       << std::setw(30) << *it 
                       << std::setw(10) << std::hex << colorcode << std::dec
                       << std::setw(15) << colorname 
                       << std::endl ; 
        }
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




