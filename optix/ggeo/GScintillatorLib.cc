#include "GScintillatorLib.hh"
#include <cassert>

#include "GItemList.hh"
#include "NPY.hpp"

#include <iomanip>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal



const char* GScintillatorLib::slow_component    = "slow_component" ;
const char* GScintillatorLib::fast_component    = "fast_component" ;


const char* GScintillatorLib::keyspec = 
"slow_component:SLOWCOMPONENT," 
"fast_component:FASTCOMPONENT," 
"reemission_cdf:DUMMY," 
;

GScintillatorLib* GScintillatorLib::load(GCache* cache)
{
    GScintillatorLib* lib = new GScintillatorLib(cache);
    lib->loadFromCache();
    return lib ; 
}

void GScintillatorLib::init()
{
    setKeyMap(keyspec);
    defineDefaults(getDefaults());
}

void GScintillatorLib::add(GPropertyMap<float>* scint)
{
    assert(!isClosed());
    m_scintillators.push_back(scint);
}

void GScintillatorLib::defineDefaults(GPropertyMap<float>* /*defaults*/)
{
}

void GScintillatorLib::import()
{
    LOG(info) << "GScintillatorLib::import "  ; 
    m_buffer->Summary();
}

NPY<float>* GScintillatorLib::createBuffer()
{
    unsigned int ni = getNumScintillators();
    unsigned int nj = m_icdf_length ;
    unsigned int nk = 1 ; 

    LOG(info) << "GScintillatorLib::createBuffer " 
              << " ni " << ni 
              << " nj " << nj 
              << " nk " << nk 
              ;  

    NPY<float>* buf = NPY<float>::make(ni, nj, nk); 
    buf->zero();
    float* data = buf->getValues();

    for(unsigned int i=0 ; i < ni ; i++)
    {
        GPropertyMap<float>* scint = m_scintillators[i] ;
        GProperty<float>* cdf = constructReemissionCDF(scint);
        assert(cdf);

        GProperty<float>* icdf = constructInvertedReemissionCDF(scint);
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


GItemList*  GScintillatorLib::createNames()
{
    unsigned int ni = getNumScintillators();
    GItemList* names = new GItemList(getType());
    for(unsigned int i=0 ; i < ni ; i++)
    {
        GPropertyMap<float>* scint = m_scintillators[i] ;
        names->add(scint->getShortName());
    }
    return names ; 
}
 


GProperty<float>* GScintillatorLib::constructInvertedReemissionCDF(GPropertyMap<float>* pmap)
{
    std::string name = pmap->getShortNameString();

    typedef GProperty<float> P ; 

    P* slow = getProperty(pmap, slow_component);
    P* fast = getProperty(pmap, fast_component);
    assert(slow != NULL && fast != NULL );


    float mxdiff = GProperty<float>::maxdiff(slow, fast);
    assert(mxdiff < 1e-6 );

    P* rrd = slow->createReversedReciprocalDomain();    // have to used reciprocal "energywise" domain for G4/NuWa agreement

    P* srrd = rrd->createZeroTrimmed();                 // trim extraneous zero values, leaving at most one zero at either extremity

    assert( srrd->getLength() == rrd->getLength() - 2); // expect to trim 2 values

    P* rcdf = srrd->createCDF();

    //
    // Why does lookup "sampling" require so many more bins to get agreeable 
    // results than standard sampling ?
    //
    // * maybe because "agree" means it matches a prior standard sampling and in
    //   the limit of many bins the techniques converge ?
    //
    // * Nope, its because of the fixed width raster across entire 0:1 in 
    //   lookup compared to "effectively" variable raster when doing value binary search
    //   as opposed to domain jump-to-the-bin : see notes in tests/GPropertyTest.cc
    //

    P* icdf = rcdf->createInverseCDF(m_icdf_length); 

    icdf->getValues()->reciprocate();  // avoid having to reciprocate lookup results : by doing it here 

    return icdf ; 
}

GProperty<float>* GScintillatorLib::constructReemissionCDF(GPropertyMap<float>* pmap)
{
    std::string name = pmap->getShortNameString();

    GProperty<float>* slow = getProperty(pmap, slow_component);
    GProperty<float>* fast = getProperty(pmap, fast_component);
    assert(slow != NULL && fast != NULL );


    float mxdiff = GProperty<float>::maxdiff(slow, fast);
    //printf("mxdiff pslow-pfast *1e6 %10.4f \n", mxdiff*1e6 );
    assert(mxdiff < 1e-6 );

    GProperty<float>* rrd = slow->createReversedReciprocalDomain();
    GProperty<float>* cdf = rrd->createCDF();
    delete rrd ; 
    return cdf ;
}




