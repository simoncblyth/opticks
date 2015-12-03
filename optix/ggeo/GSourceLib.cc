#include "GSourceLib.hh"
#include "GSource.hh"
#include "GItemList.hh"

#include <cassert>

#include "NPY.hpp"
#include "NLog.hpp"

const unsigned int GSourceLib::icdf_length = 1024 ; 
const char* GSourceLib::radiance_ = "radiance" ; 


void GSourceLib::save()
{
    saveToCache();
}

GSourceLib* GSourceLib::load(GCache* cache)
{
    GSourceLib* lib = new GSourceLib(cache);
    lib->loadFromCache();
    return lib ; 
}

void GSourceLib::init()
{
    defineDefaults(getDefaults());
}

void GSourceLib::add(GSource* source)
{
    assert(!isClosed());
    m_source.push_back(source);
}



void GSourceLib::defineDefaults(GPropertyMap<float>* /*defaults*/)
{
    LOG(debug) << "GSourceLib::defineDefaults"  ; 
}
void GSourceLib::sort()
{
    LOG(debug) << "GSourceLib::sort"  ; 
}
void GSourceLib::import()
{
    LOG(debug) << "GSourceLib::import "  ; 
}


void GSourceLib::generateBlackBodySample(unsigned int n)
{
    GSource* source = GSource::make_blackbody_source("D65", 0, 6500.f );    
    GProperty<float>* icdf = constructInvertedSourceCDF(source);
    GAry<float>* sample = icdf->lookupCDF(n);     
    sample->save("/tmp/blackbody.npy");
}



NPY<float>* GSourceLib::createBuffer()
{
    if(getNumSources() == 0)
    {
        LOG(info) << "GSourceLib::createBuffer adding standard source " ;
        GSource* source = GSource::make_blackbody_source("D65", 0, 6500.f );    
        add(source);
    } 


    unsigned int ni = getNumSources();
    unsigned int nj = icdf_length ;
    unsigned int nk = 1 ; 

    LOG(debug) << "GSourceLib::createBuffer " 
              << " ni " << ni 
              << " nj " << nj 
              << " nk " << nk 
              ;  

    NPY<float>* buf = NPY<float>::make(ni, nj, nk); 
    buf->zero();
    float* data = buf->getValues();

    for(unsigned int i=0 ; i < ni ; i++)
    {
        GPropertyMap<float>* source = m_source[i] ;
        GProperty<float>* cdf = constructSourceCDF(source);
        assert(cdf);

        GProperty<float>* icdf = constructInvertedSourceCDF(source);
        assert(icdf);
        assert(icdf->getLength() == nj);

        for( unsigned int j = 0; j < nj ; ++j ) 
        {
            unsigned int offset = i*nj*nk + j*nk ;  
            data[offset+0] = icdf->getValue(j);
        }
   } 
   return buf ; 
}


GItemList*  GSourceLib::createNames()
{
    unsigned int ni = getNumSources();
    GItemList* names = new GItemList(getType());
    for(unsigned int i=0 ; i < ni ; i++)
    {
        GPropertyMap<float>* source = m_source[i] ;
        names->add(source->getShortName());
    }
    return names ; 
}
 


GProperty<float>* GSourceLib::constructInvertedSourceCDF(GPropertyMap<float>* pmap)
{
    typedef GProperty<float> P ; 

    P* radiance = pmap->getProperty(radiance_);
    assert(radiance);

    P* dist = radiance->createZeroTrimmed();  // trim any extraneous zero values, leaving at most one zero at either extremity

    P* cdf = dist->createCDF();

    P* icdf = cdf->createInverseCDF(icdf_length); 

    return icdf ; 
}

GProperty<float>* GSourceLib::constructSourceCDF(GPropertyMap<float>* pmap)
{
    typedef GProperty<float> P ; 

    P* radiance = pmap->getProperty(radiance_);
    assert(radiance);

    P* dist = radiance->createZeroTrimmed();  // trim any extraneous zero values, leaving at most one zero at either extremity

    P* cdf = dist->createCDF();

    return cdf ;
}


