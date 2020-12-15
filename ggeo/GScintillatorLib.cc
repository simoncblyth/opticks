/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include <cassert>
#include "NPY.hpp"
#include "BMeta.hh"

#include "GDomain.hh"
#include "GAry.hh"
#include "GProperty.hh"
#include "GPropertyMap.hh"
#include "GItemList.hh"
#include "GScintillatorLib.hh"

#include "PLOG.hh"


const plog::Severity GScintillatorLib::LEVEL = debug ; 

const char* GScintillatorLib::slow_component    = "slow_component" ;
const char* GScintillatorLib::fast_component    = "fast_component" ;


const char* GScintillatorLib::keyspec = 
"slow_component:SLOWCOMPONENT," 
"fast_component:FASTCOMPONENT," 
"reemission_cdf:DUMMY," 
;

void GScintillatorLib::Summary(const char* msg)
{
   LOG(info) << msg 
             << " num_scintillators " << getNumRaw() 
             ;
}

void GScintillatorLib::dump(const char* msg)
{
   LOG(info) << msg 
             << " num_scintillators " << getNumRaw() 
             ;

   dumpRaw(msg); 
}

void GScintillatorLib::save()
{
    saveToCache();
    saveRaw();
}

GScintillatorLib* GScintillatorLib::load(Opticks* ok)
{
    GScintillatorLib* lib = new GScintillatorLib(ok);
    lib->loadFromCache();
    lib->loadRaw();
    return lib ; 
}


GScintillatorLib::GScintillatorLib( Opticks* ok, unsigned int icdf_length) 
    :
    GPropertyLib(ok, "GScintillatorLib", true ),
    m_icdf_length(icdf_length)
{
    init();
}

unsigned int GScintillatorLib::getNumScintillators()
{
    return getNumRaw();
}




void GScintillatorLib::init()
{
    setKeyMap(keyspec);
    defineDefaults(getDefaults());
}

void GScintillatorLib::add(GPropertyMap<float>* scint)
{
    assert(!isClosed());
    addRaw(scint);
}

void GScintillatorLib::defineDefaults(GPropertyMap<float>* /*defaults*/)
{
    LOG(debug) << "GScintillatorLib::defineDefaults"  ; 
}
void GScintillatorLib::sort()
{
    LOG(debug) << "GScintillatorLib::sort"  ; 
}
void GScintillatorLib::import()
{
    LOG(debug) << "GScintillatorLib::import "  ; 
    //m_buffer->Summary("GScintillatorLib::import");
}

BMeta* GScintillatorLib::createMeta()
{
    return NULL ; 
}


NPY<float>* GScintillatorLib::createBuffer()
{
    unsigned int ni = getNumRaw();
    unsigned int nj = m_icdf_length ;
    unsigned int nk = 1 ; 

    LOG(LEVEL) 
          << " ni " << ni 
          << " nj " << nj 
          << " nk " << nk 
          ;  

    NPY<float>* buf = NPY<float>::make(ni, nj, nk); 
    buf->zero();
    float* data = buf->getValues();

    for(unsigned int i=0 ; i < ni ; i++)
    {
        GPropertyMap<float>* scint = getRaw(i) ;
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
    unsigned int ni = getNumRaw();
    GItemList* names = new GItemList(getType());
    for(unsigned int i=0 ; i < ni ; i++)
    {
        GPropertyMap<float>* scint = getRaw(i) ;
        names->add(scint->getShortName());
    }
    return names ; 
}


/**
GScintillatorLib::constructInvertedReemissionCDF
---------------------------------------------------

This is invoked by the GScintillatorLib::createBuffer method 
above with the results being persisted to the buffer.



Why does lookup "sampling" require so many more bins to get agreeable 
results than standard sampling ?

* maybe because "agree" means it matches a prior standard sampling and in
  the limit of many bins the techniques converge ?
  
* Nope, its because of the fixed width raster across entire 0:1 in 
  lookup compared to "effectively" variable raster when doing value binary search
  as opposed to domain jump-to-the-bin : see notes in tests/GPropertyTest.cc

**/

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

    unsigned int l_srrd = srrd->getLength() ;
    unsigned int l_rrd = rrd->getLength()  ;

    if( l_srrd != l_rrd - 2)
    {
       LOG(debug) 
           << "was expecting to trim 2 values "
           << " l_srrd " << l_srrd 
           << " l_rrd " << l_rrd 
           ;
    }
    //assert( l_srrd == l_rrd - 2); // expect to trim 2 values

    P* rcdf = srrd->createCDF();

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


