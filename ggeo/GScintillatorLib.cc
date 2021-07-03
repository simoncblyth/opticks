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
#include "GMaterialLib.hh"
#include "GMaterial.hh"


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
    saveRawEnergy();
}

GScintillatorLib* GScintillatorLib::load(Opticks* ok)
{
    GScintillatorLib* lib = new GScintillatorLib(ok);
    lib->loadFromCache();
    lib->loadRaw();
    lib->loadRawEnergy();
    return lib ; 
}


GScintillatorLib::GScintillatorLib( Opticks* ok, unsigned icdf_length) 
    :
    GPropertyLib(ok, "GScintillatorLib", true ),
    m_mlib(nullptr),
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

void GScintillatorLib::add(GPropertyMap<double>* scint)
{
    assert(!isClosed());
    addRaw(scint);
}

void GScintillatorLib::defineDefaults(GPropertyMap<double>* /*defaults*/)
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


/**
GScintillatorLib::createBuffer
--------------------------------

This gets called from the base in GPropertyLib::close after 
which GPropertyLib::setBuffer is used. 

**/

NPY<double>* GScintillatorLib::createBuffer()
{
    return m_g4icdf ? m_g4icdf : legacyCreateBuffer() ; 
}

GItemList*  GScintillatorLib::createNames()
{
    return m_g4icdf ? geant4ICDFCreateNames() : legacyCreateNames() ;  
}


GItemList*  GScintillatorLib::geant4ICDFCreateNames() const 
{
    unsigned ni = 1 ; // new approach currently limited to only 1 scintillator material
    GItemList* names = new GItemList(getType());
    for(unsigned int i=0 ; i < ni ; i++)
    {
        std::string name =  m_g4icdf->getMeta<std::string>("name", "" ); 
        bool empty = name.empty();  
        if(empty) LOG(fatal) << "Geant4ICDF must have non-empty name metadata " ; 
        assert( !empty ); 
        names->add(name.c_str());
    }
    return names ; 
}

GItemList*  GScintillatorLib::legacyCreateNames() const 
{
    unsigned ni = getNumRaw();
    GItemList* names = new GItemList(getType());
    for(unsigned int i=0 ; i < ni ; i++)
    {
        GPropertyMap<double>* scint = getRaw(i) ;
        names->add(scint->getShortName());
    }
    return names ; 
}



void GScintillatorLib::setGeant4InterpolatedICDF( NPY<double>* g4icdf )
{
    m_g4icdf = g4icdf ; 
}
NPY<double>* GScintillatorLib::getGeant4InterpolatedICDF() const 
{
    return m_g4icdf ; 
}




/**
GScintillatorLib::legacyCreateBuffer
----------------------------------------

The legacy approach was implemenented in the era of Opticks using an export/import approach 
which made it necessary to reimplement many things that are easier in the current 
approach to just steal from Geant4.

**/

NPY<double>* GScintillatorLib::legacyCreateBuffer() const 
{
    LOG(fatal) << " using legacy approach, avoid this by GScintillatorLib::setGeant4InterpolatedICDF  " ; 
    unsigned int ni = getNumRaw();
    unsigned int nj = m_icdf_length ;
    unsigned int nk = 1 ; 

    LOG(LEVEL) 
          << " ni " << ni 
          << " nj " << nj 
          << " nk " << nk 
          ;  

    NPY<double>* buf = NPY<double>::make(ni, nj, nk); 
    buf->zero();
    double* data = buf->getValues();

    for(unsigned int i=0 ; i < ni ; i++)
    {
        GPropertyMap<double>* scint = getRaw(i) ;
        GProperty<double>* cdf = legacyConstructReemissionCDF(scint);
        assert(cdf);

        GProperty<double>* icdf = legacyConstructInvertedReemissionCDF(scint);
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



/**
GScintillatorLib::legacyConstructInvertedReemissionCDF
---------------------------------------------------------

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

GProperty<double>* GScintillatorLib::legacyConstructInvertedReemissionCDF(GPropertyMap<double>* pmap) const 
{
    std::string name = pmap->getShortNameString();

    typedef GProperty<double> P ; 

    P* slow = getProperty(pmap, slow_component);
    P* fast = getProperty(pmap, fast_component);
    assert(slow != NULL && fast != NULL );


    double mxdiff = GProperty<double>::maxdiff(slow, fast);
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

GProperty<double>* GScintillatorLib::legacyConstructReemissionCDF(GPropertyMap<double>* pmap) const 
{
    std::string name = pmap->getShortNameString();

    GProperty<double>* slow = getProperty(pmap, slow_component);
    GProperty<double>* fast = getProperty(pmap, fast_component);
    assert(slow != NULL && fast != NULL );

    double mxdiff = GProperty<double>::maxdiff(slow, fast);
    //printf("mxdiff pslow-pfast *1e6 %10.4f \n", mxdiff*1e6 );
    assert(mxdiff < 1e-6 );

    GProperty<double>* rrd = slow->createReversedReciprocalDomain();
    GProperty<double>* cdf = rrd->createCDF();
    delete rrd ; 
    return cdf ;
}





/**
GScintillatorLib::prepare
---------------------------

Currently invoked from GGeo::prepare/GGeo::prepareScintillatorLib 

TODO: move to invoking it earlier, eg in a new X4PhysicalVolume::convertScintillators invoked from X4PhysicalVolume::init

1. collect scintillator raw materials from GMaterialLib into *m_scintillators_raw* identified by the 
   presence of three properties : SLOWCOMPONENT,FASTCOMPONENT,REEMISSIONPROB 

2. GPropertyLib::addRaw the scintillator property maps  

3. GPropertyLib::close the lib which invokes GScintillatorLib::createBuffer and sets the buffer


**/


/**

void GScintillatorLib::prepare()
{
    LOG(LEVEL); 

    m_mlib = GMaterialLib::Get(); 

    assert( m_mlib ) ; 

    const char* props = "SLOWCOMPONENT,FASTCOMPONENT,REEMISSIONPROB" ;
 
    m_scintillators_raw = m_mlib->getRawMaterialsWithProperties(props, ',' ); 

    unsigned int num_scint = m_scintillators_raw.size() ; 

    if(num_scint == 0)
    {
        LOG(LEVEL) << " found no scintillator materials  " ; 
    }
    else
    {
        LOG(LEVEL) << " found " << num_scint << " scintillator materials  " ; 

        for(unsigned int i=0 ; i < num_scint ; i++)
        {
            GPropertyMap<double>* scint = dynamic_cast<GPropertyMap<double>*>(getScintillatorMaterial(i));  
            add(scint);
        }

        close(); 
    }
}

void GScintillatorLib::dumpScintillatorMaterials(const char* msg)
{
    LOG(info)<< msg ;
    for(unsigned int i=0; i<m_scintillators_raw.size() ; i++)
    {
        GMaterial* mat = m_scintillators_raw[i];
        //mat->Summary();
        std::cout << std::setw(30) << mat->getShortName()
                  << " keys: " << mat->getKeysString()
                  << std::endl ; 
    }              
}

unsigned int GScintillatorLib::getNumScintillatorMaterials()
{
    return m_scintillators_raw.size();
}

GMaterial* GScintillatorLib::getScintillatorMaterial(unsigned int index)
{
    return index < m_scintillators_raw.size() ? m_scintillators_raw[index] : NULL ; 
}


**/


