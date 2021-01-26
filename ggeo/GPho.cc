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

#include <iostream>
#include <iomanip>

#include "BStr.hh"
#include "NGLM.hpp"
#include "GLMFormat.hpp"
#include "GLMPrint.hpp"
#include "NPY.hpp"
#include "NSlice.hpp"

#include "OpticksPhoton.h"

#include "GGeo.hh"
#include "GPho.hh"
#include "GVolume.hh"

#include "PLOG.hh"

/**
Started from npy/NPho adding use of GGeo for access to transforms
**/

const plog::Severity GPho::LEVEL = PLOG::EnvLevel("GPho", "DEBUG"); 

const char* GPho::A = "All" ; 
const char* GPho::L = "Landed on sensors" ; 
const char* GPho::H = "Hits" ; 

void GPho::setSelection(char selection)
{
    assert( selection == 'A' || selection == 'L' || selection == 'H' ); 
    m_selection = selection ; 
}

const char* GPho::getSelectionName() const 
{
    const char* name = NULL ; 
    switch(m_selection)
    {
        case 'A': name = A ; break ; 
        case 'L': name = L ; break ; 
        case 'H': name = H ; break ; 
    }
    return name ; 
}

const char* GPho::DEFAULT_OPT = "nidx,nrpo,post,lpst,okfl" ; 

GPho::GPho(const GGeo* ggeo, const char* opt) 
    :  
    m_photons(NULL),
    m_msk(NULL),
    m_num_photons(0),
    m_ggeo(ggeo),
    m_opt(opt ? strdup(opt) : DEFAULT_OPT ),
    m_selection('A')
{
    setOpt(m_opt); 
}

void GPho::setOpt(const char* opt)
{
    m_opt = opt ? strdup(opt) : DEFAULT_OPT ; 
}

/**
GPho::setPhotons
------------------

The way array is only available when optickscore/OpticksSwitches.h:WITH_WAY_BUFFER is defined. 
When there us no *way* use NULL for the argmument.

The way array provides extra information about the photons such as the original 
start time and a boundary_pos at a configurable border along the path of the photon.

**/

void GPho::setPhotons(const  NPY<float>* photons)
{
    m_photons = photons ; 

    const void* aux = photons->getAux(); 
    const NPY<float>* way = aux ? static_cast<const NPY<float>*>(aux) : NULL ; 
    m_way = way ; 

    assert( m_photons ); 
    assert( m_photons->hasShape(-1,4,4) );
    m_num_photons = m_photons->getNumItems() ;

    if(m_way)
    {
        assert( m_way->hasShape(m_num_photons, 2, 4)); 
        bool dump = false ;
        unsigned mismatch = wayConsistencyCheck(dump); 
        if(mismatch > 0 ) wayConsistencyCheck(true) ; 
        assert( mismatch == 0 ); 
    }

    m_msk = m_photons->getMsk();  
    if(m_msk)
    {
        unsigned num_origin_indices = m_msk->getNumItems() ; 
        assert( num_origin_indices == m_num_photons && "associated masks are required to have been already applied to the photons" ); 
    }
}

/**
GPho::wayConsistencyCheck
--------------------------

Check consistency of the flags and index between the *photon* and *way* arrays. 

**/

unsigned GPho::wayConsistencyCheck(bool dump)
{
    assert(m_photons);     
    assert(m_way);     

    int j_flags = -1 ;  // last quad
    int k_flags = 3 ;  // last element of last quad (-1 would work too)

    int j_index = -1 ;  // last quad 
    int k_index = 2 ;  // penultimate element of last quad (-2 would work fine too)

    unsigned mismatch = 0 ; 
    unsigned mismatch_flags = NPY<float>::compare_element_jk( m_photons, m_way, j_flags, k_flags, dump ); 
    unsigned mismatch_index = NPY<float>::compare_element_jk( m_photons, m_way, j_index, k_index, dump ); 
    
    mismatch += mismatch_flags ; 
    mismatch += mismatch_index ; 

    LOG(info) 
        << " mismatch_flags " << mismatch_flags
        << " mismatch_index " << mismatch_index
        ;

    return mismatch ; 
}






unsigned GPho::getNumPhotons() const 
{
    return m_num_photons ; 
}
const NPY<float>* GPho::getPhotons() const 
{
    return m_photons ; 
}
const NPY<float>* GPho::getWay() const 
{
    return m_way ; 
}

bool GPho::hasWay() const 
{
    return m_way != NULL ; 
}
/**
GPho::getWayPositionTime
-------------------------

Will assert if m_way is NULL, check before calling with hasWay.

**/
glm::vec4 GPho::getWayPositionTime(unsigned i) const 
{
    assert( m_way ); 
    glm::vec4 q0 = m_way->getQuad_(i,0);
    return q0 ; 
}
/**
GPho::getWayOriginTime
------------------------

Will assert if m_way is NULL, check before calling with hasWay.

**/
float GPho::getWayOriginTime(unsigned i) const 
{
    assert( m_way ); 
    glm::vec4 q1 = m_way->getQuad_(i,1);
    return q1.x ; 
}

/**
GPho::getWayOriginTrackID
----------------------------

Will assert if m_way is NULL, check before calling with hasWay.

**/
int GPho::getWayOriginTrackID(unsigned i) const 
{
    assert( m_way ); 
    glm::ivec4 q1 = m_way->getQuadI(i,1);
    return q1.y ; 
}







glm::vec4 GPho::getPositionTime(unsigned i) const 
{
    glm::vec4 post = m_photons->getQuad_(i,0);
    return post ; 
}

glm::vec4 GPho::getDirectionWeight(unsigned i) const 
{
    glm::vec4 dirw = m_photons->getQuad_(i,1);
    return dirw ; 
}
glm::vec4 GPho::getPolarizationWavelength(unsigned i) const 
{
    glm::vec4 polw = m_photons->getQuad_(i,2);
    return polw ; 
}
glm::ivec4 GPho::getFlags(unsigned i) const 
{
    glm::ivec4 flgs = m_photons->getQuadI(i,3);  // union type shifting getter
    return flgs ; 
}

/**
GPho::getOpticksPhotonFlags
---------------------------

The float flags contain the bits of unsigned and signed integers with some bit packing.  
These are decoded using OpticksPhotonFlags.

**/

OpticksPhotonFlags GPho::getOpticksPhotonFlags(unsigned i) const 
{
    glm::vec4 flgs = m_photons->getQuad_(i,3);  
    OpticksPhotonFlags okfl(flgs); 
    return okfl ;  
}  

int GPho::getLastIntersectBoundary(unsigned i) const
{
    glm::vec4 flgs = m_photons->getQuad_(i,3); 
    return OpticksPhotonFlags::Boundary(flgs);  
} 

/**
GPho::getLastIntersectNodeIndex
----------------------------------

This is now using the NodeIndex from the flags.
FORMERLY stomped on the weight with the nidx.

**/
unsigned GPho::getLastIntersectNodeIndex(unsigned i) const
{
    glm::vec4 flgs = m_photons->getQuad_(i,3); 
    return OpticksPhotonFlags::NodeIndex(flgs);  
}
glm::uvec4 GPho::getLastIntersectNRPO(unsigned i) const 
{
    unsigned nidx = getLastIntersectNodeIndex(i); 
    glm::uvec4 nrpo = m_ggeo->getNRPO(nidx); 
    return nrpo ; 
}

glm::mat4 GPho::getLastIntersectTransform(unsigned i) const 
{
    unsigned nidx = getLastIntersectNodeIndex(i);
    glm::mat4 tr = m_ggeo->getTransform(nidx); 
    return tr ; 
}
glm::mat4 GPho::getLastIntersectInverseTransform(unsigned i) const 
{
    unsigned nidx = getLastIntersectNodeIndex(i);
    glm::mat4 it = m_ggeo->getInverseTransform(nidx); 
    return it ; 
}

glm::vec4 GPho::getLocalPositionTime(unsigned i) const 
{
    glm::vec4 post = getPositionTime(i); 
    float time = post.w ; 
    post.w = 1.f ;  // w=1. to transform as a position  
    glm::mat4 it = getLastIntersectInverseTransform(i); 
    glm::vec4 lpost = it * post ; 
    lpost.w = time ; 
    return lpost ; 
}
glm::vec4 GPho::getLocalDirectionWeight(unsigned i) const 
{
    glm::vec4 dirw = getDirectionWeight(i); 
    float weight = dirw.w ;  
    dirw.w = 0.f ;     // w=0. to transform as a vector   
    glm::mat4 it = getLastIntersectInverseTransform(i); 
    glm::vec4 ldrw = it * dirw ; 
    ldrw.w = weight ; 
    return ldrw ; 
}
glm::vec4 GPho::getLocalPolarizationWavelength(unsigned i) const 
{
    glm::vec4 polw = getPolarizationWavelength(i); 
    float wavelength = polw.w ;   
    polw.w = 0.f ;   // w=0. to transform as a vector   
    glm::mat4 it = getLastIntersectInverseTransform(i); 
    glm::vec4 lpow = it * polw ; 
    lpow.w = wavelength ; 
    return lpow ; 
}



NPY<float>* GPho::makeLocalPhotons() const 
{
    NPY<float>* lpho = NPY<float>::make(m_num_photons, 4, 4); 
    lpho->zero(); 
    for(unsigned i=0 ; i < m_num_photons ; i++)
    { 
        glm::vec4 lpst = getLocalPositionTime(i);
        glm::vec4 ldrw = getLocalDirectionWeight(i);
        glm::vec4 lpow = getLocalPolarizationWavelength(i);
        glm::ivec4 flgs = getFlags(i);

        lpho->setQuad_(lpst, i, 0); 
        lpho->setQuad_(ldrw, i, 1); 
        lpho->setQuad_(lpow, i, 2); 
        lpho->setQuadI(flgs, i, 3); 
    } 
    return lpho ; 
}

void GPho::saveLocalPhotons(const char* path) const 
{
    NPY<float>* lpho = makeLocalPhotons(); 
    LOG(info) << " path " << path ; 
    lpho->save(path); 
}


std::string GPho::desc() const 
{
    std::stringstream ss ;
    ss << "GPho " 
       << " ph " << ( m_photons ? m_photons->getShapeString() : "-" ) 
       << " wy " << ( m_way ? m_way->getShapeString() : "-" ) 
       ; 
    return ss.str();
}


bool GPho::isLandedOnSensor(unsigned i) const 
{
    OpticksPhotonFlags okfl = getOpticksPhotonFlags(i); 
    return okfl.sensorIndex != GVolume::SENSOR_UNSET ;  
}
bool GPho::isHit(unsigned i) const 
{
    glm::ivec4 flgs = getFlags(i);
    unsigned pflg = flgs.w ; 
    return ( pflg & SURFACE_DETECT) != 0 ; 
}


std::string GPho::desc(unsigned i) const 
{
    glm::vec4 post = getPositionTime(i);
    glm::vec4 dirw = getDirectionWeight(i);
    glm::vec4 polw = getPolarizationWavelength(i);
    glm::ivec4 flgs = getFlags(i);

    unsigned nidx = getLastIntersectNodeIndex(i); 
    glm::uvec4 nrpo = getLastIntersectNRPO(i); 

    glm::vec4 lpst = getLocalPositionTime(i);
    glm::vec4 ldrw = getLocalDirectionWeight(i);
    glm::vec4 lpow = getLocalPolarizationWavelength(i);

    OpticksPhotonFlags okfl = getOpticksPhotonFlags(i); 
    assert( nidx == okfl.nodeIndex ); 


    bool _nidx = BStr::Contains(m_opt, "nidx", ',' ) ;
    bool _nrpo = BStr::Contains(m_opt, "nrpo", ',' ) ;
    bool _mski = BStr::Contains(m_opt, "mski", ',' ) ;
    bool _post = BStr::Contains(m_opt, "post", ',' ) ;
    bool _lpst = BStr::Contains(m_opt, "lpst", ',' ) ;
    bool _ldrw = BStr::Contains(m_opt, "ldrw", ',' ) ;
    bool _lpow = BStr::Contains(m_opt, "lpow", ',' ) ;
    bool _dirw = BStr::Contains(m_opt, "dirw", ',' ) ;
    bool _polw = BStr::Contains(m_opt, "polw", ',' ) ;
    bool _flgs = BStr::Contains(m_opt, "flgs", ',' ) ;
    bool _okfl = BStr::Contains(m_opt, "okfl", ',' ) ;
    bool _way0 = BStr::Contains(m_opt, "way0", ',' ) ;
    bool _way1 = BStr::Contains(m_opt, "way1", ',' ) ;

    std::stringstream ss ;
    ss << " i " << std::setw(7) << i ; 
    if(_mski) ss << " mski " << std::setw(7) << m_photons->getMskIndex(i)  ; 
    if(_nidx) ss << " nidx " << std::setw(7) << nidx ; 
    if(_nrpo) ss << " nrpo " << std::setw(20) << gpresent(nrpo,7,3,4,3) ; ; 
    if(_post) ss << " post " << std::setw(20) << gpresent(post) ;
    if(_lpst) ss << " lpst " << std::setw(20) << gpresent(lpst) ;
    if(_ldrw) ss << " ldrw " << std::setw(20) << gpresent(ldrw) ;
    if(_lpow) ss << " lpow " << std::setw(20) << gpresent(lpow) ;
    if(_dirw) ss << " dirw " << std::setw(20) << gpresent(dirw) ;
    if(_polw) ss << " polw " << std::setw(20) << gpresent(polw) ;
    if(_flgs) ss << " flgs " << std::setw(20) << gpresent(flgs) ;
    if(_okfl) ss << " okfl " << std::setw(20) << okfl.brief() ;

    if(m_way)
    {
         _way0 = true ; 
         _way1 = true ; 
         glm::vec4  way0 = m_way->getQuad_(i, 0); 
         glm::uvec4 way1 = m_way->getQuadU(i, 1); 
         if(_way0) ss << " way0 " << std::setw(20) << gpresent(way0) ; 
         if(_way1) ss << " way1 " << std::setw(20) << gpresent(way1) ; 
    }

    return ss.str();
}

void GPho::dump(unsigned modulo, unsigned margin, const char* msg) const
{
    NSlice slice(0, getNumPhotons()) ;

    LOG(info) 
        << msg 
        << " slice " << slice.description()
        << " modulo " << modulo
        << " margin " << margin 
        << " desc " << desc() 
        ; 

    for(unsigned i=slice.low ; i < slice.high ; i += slice.step )
    {
        if(slice.isMargin(i, margin) || i % modulo == 0)
        {
            std::cout << desc(i) << std::endl ; 
        }
    }
}

void GPho::dump(const char* msg, unsigned maxDump) const 
{
    unsigned numPhotons = getNumPhotons() ;
    unsigned numDump = maxDump > 0 ?  std::min( numPhotons, maxDump ) : numPhotons ; 
    LOG(info) 
        << msg 
        << " desc " << desc() 
        << " numPhotons " << numPhotons
        << " maxDump " << maxDump
        << " numDump " << numDump
        ; 

    LOG(info)
        << " selectionName " << getSelectionName()
        << " opt " << m_opt 
        ;

    unsigned count = 0 ; 
    for(unsigned i=0 ; i < numDump  ; i++)
    {
        bool select = true ; 
        switch(m_selection)
        {
           case 'A': select = true                ; break ; 
           case 'L': select = isLandedOnSensor(i) ; break ; 
           case 'H': select = isHit(i)            ; break ; 
        } 
        if(select)
        {
            count += 1 ; 
            std::cout << desc(i) << std::endl ;
        }
    }

    LOG(info)
        << " selectionName " << getSelectionName()
        << " opt " << m_opt 
        << " count " << count 
        ;
}

void GPho::Dump(const NPY<float>* ox, const GGeo* ggeo, unsigned modulo, unsigned margin, const char* opt)  // static
{
    LOG(info) << opt
              << " modulo " << modulo
              << " margin " << margin
              << " ox " << ( ox ? "Y" : "NULL" ) 
              ;
 
    if(!ox) return ; 

    GPho ph(ggeo, opt) ;
    ph.setPhotons(ox);
    ph.dump(modulo, margin); 
}

void GPho::Dump(const NPY<float>* ox, const GGeo* ggeo, unsigned maxDump, const char* opt)   // static 
{
    LOG(info) << opt
              << " ox " << ( ox ? "Y" : "NULL" ) 
              ;
 
    if(!ox) return ; 
    GPho ph(ggeo, opt) ;
    ph.setPhotons(ox);
    ph.dump("GPho::Dump", maxDump); 
}



