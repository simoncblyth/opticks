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

#include <iomanip>
#include <sstream>
#include <numeric>

#include "NLookup.hpp"
#include "NPY.hpp"
#include "CGenstepCollector.hh"

#include "OpticksHub.hh"
#include "OpticksGenstep.hh"
#include "OpticksActionControl.hh"
#include "OpticksGenstep.h"
#include "OpticksPhoton.h"

#include "PLOG.hh"



const plog::Severity CGenstepCollector::LEVEL = PLOG::EnvLevel("CGenstepCollector", "DEBUG")  ;

CGenstepCollector* CGenstepCollector::INSTANCE = NULL ;

CGenstepCollector* CGenstepCollector::Instance()
{
   assert(INSTANCE && "CGenstepCollector has not been instanciated");
   return INSTANCE ;
}

CGenstepCollector::CGenstepCollector(const NLookup* lookup)    
    :
    m_lookup(lookup),
    m_genstep(NPY<float>::make(0,6,4)),
    m_genstep_itemsize(m_genstep->getNumValues(1)),
    m_genstep_values(new float[m_genstep_itemsize]),
    m_scintillation_count(0),
    m_cerenkov_count(0),
    m_torch_count(0),
    m_torch_emitsource_count(0),
    m_machinery_count(0)
{
    assert( m_genstep_itemsize == 6*4 );

    bool lookup_complete = m_lookup && m_lookup->isClosed() ;
    if(!lookup_complete)
    {
       LOG(error)
          << " lookup is not complete : will not be able to collect real gensteps, only machinery ones " ;  
    }
    //assert( lookup_complete ); 
    INSTANCE = this ; 
}


void CGenstepCollector::reset()
{
    m_scintillation_count = 0 ; 
    m_cerenkov_count = 0 ; 
    m_torch_count = 0 ; 
    m_torch_emitsource_count = 0 ; 
    m_machinery_count = 0 ; 
    m_genstep->reset(); 
    m_gs_photons.clear(); 

}
void CGenstepCollector::save(const char* path)
{
    m_genstep->save(path); 
}
void CGenstepCollector::load(const char* path)
{
    reset(); 
    m_genstep = NPY<float>::load(path);   
    import(); 
}


/**
The below assume 1-to-1 between eventId and genstep arrays, 
intend to break this assumption in future.
**/

void CGenstepCollector::setArrayContentIndex(unsigned eventId)
{
    m_genstep->setArrayContentIndex(eventId); 
}
unsigned CGenstepCollector::getArrayContentIndex() const 
{
    return m_genstep->getArrayContentIndex(); 
}


void CGenstepCollector::import()
{
    unsigned ni = m_genstep->getNumItems() ;

    assert( m_scintillation_count == 0);
    assert( m_cerenkov_count == 0);
    assert( m_torch_count == 0);
    assert( m_machinery_count == 0);

    for(unsigned i=0 ; i < ni ; i++)
    {
        unsigned gentype = m_genstep->getInt(i,0u,0u);
        unsigned numPhotons = m_genstep->getInt(i,0u,3u);

        if(OpticksGenstep::IsScintillation(gentype))  m_scintillation_count += 1 ;       
        else if(OpticksGenstep::IsCerenkov(gentype))  m_cerenkov_count += 1 ;       
        else if(OpticksGenstep::IsTorchLike(gentype))  m_torch_count += 1 ;       
        else if(OpticksGenstep::IsMachinery(gentype)) m_machinery_count += 1 ;       

        m_gs_photons.push_back(numPhotons); 
    }

    unsigned total = m_scintillation_count + m_cerenkov_count + m_torch_count + m_machinery_count ; 
    assert( total == ni ); 
}




/**
CGenstepCollector::translate
-----------------------------

Uses the lookup to translate Geant4 material index into 
GBndLib material line for use with GPU texture.

**/
int CGenstepCollector::translate(int acode) const 
{
    assert( m_lookup && m_lookup->isClosed() ); 
    int bcode = m_lookup->a2b(acode) ;
    return bcode ; 
}


unsigned CGenstepCollector::getNumGensteps() const 
{
    return m_gs_photons.size(); 
}
unsigned CGenstepCollector::getNumPhotons() const 
{
    return std::accumulate(m_gs_photons.begin(), m_gs_photons.end(), 0u );
} 
unsigned CGenstepCollector::getNumPhotons(unsigned gs_idx) const 
{
    assert( gs_idx < m_gs_photons.size() ); 
    return m_gs_photons[gs_idx] ; 
} 




NPY<float>*  CGenstepCollector::getGensteps() const 
{
    consistencyCheck() ;
    return m_genstep ; 
}


void CGenstepCollector::consistencyCheck() const 
{
     unsigned numItems = m_genstep->getNumItems();
     bool consistent = numItems == m_scintillation_count + m_cerenkov_count + m_torch_count + m_machinery_count ;
     if(!consistent)
         LOG(fatal) << "CGenstepCollector::consistencyCheck FAIL " 
                    << description()
                    ;
     assert(consistent);
}


std::string CGenstepCollector::desc() const
{
    std::stringstream ss ; 
    ss 
       << " ngs " << std::setw(3) << m_genstep->getNumItems() 
       << " nsc " << std::setw(3) << m_scintillation_count
       << " nck " << std::setw(3) << m_cerenkov_count
       << " nto " << std::setw(3) << m_torch_count
       << " nma " << std::setw(3) << m_machinery_count
       << " tot " << std::setw(3) << m_scintillation_count + m_cerenkov_count + m_torch_count + m_machinery_count 
       ;
    return ss.str();
}

std::string CGenstepCollector::description() const
{
    std::stringstream ss ; 
    ss << " CGenstepCollector "
       << " numItems " << m_genstep->getNumItems() 
       << " scintillation_count " << m_scintillation_count
       << " cerenkov_count " << m_cerenkov_count
       << " torch_count " << m_torch_count
       << " machinery_count " << m_machinery_count
       << " step_count " << m_scintillation_count + m_cerenkov_count + m_torch_count + m_machinery_count 
       ;
    return ss.str();
}

void CGenstepCollector::Summary(const char* msg) const 
{ 
    LOG(info) << msg 
              << description()
              ;
}

void CGenstepCollector::setReservation(int items)
{
    m_genstep->setReservation(items); 
}
int CGenstepCollector::getReservation() const 
{
    return m_genstep->getReservation();
}

void CGenstepCollector::collectScintillationStep
(
            G4int                gentype, 
            G4int                parentId,
            G4int                materialId,
            G4int                numPhotons,
            
            G4double             x0_x,  
            G4double             x0_y,  
            G4double             x0_z,  
            G4double             t0, 

            G4double             deltaPosition_x, 
            G4double             deltaPosition_y, 
            G4double             deltaPosition_z, 
            G4double             stepLength, 

            G4int                pdgCode, 
            G4double             pdgCharge, 
            G4double             weight, 
            G4double             meanVelocity, 

            G4int                scntId,      
            G4double             slowerRatio,
            G4double             slowTimeConstant,
            G4double             slowerTimeConstant,

            G4double             scintillationTime,
            G4double             scintillationIntegralMax,
            G4double             spare1,
            G4double             spare2
) 
{
     m_scintillation_count += 1 ;   // 1-based index
     m_gs_photons.push_back(numPhotons); 

     LOG(LEVEL)
          << " gentype " << gentype
          << " gentype " << OpticksGenstep::Gentype(gentype)
          << " pdgCode " << pdgCode
          << " numPhotons " << std::setw(4) << numPhotons 
          << desc()
          ;

     assert( OpticksGenstep::IsScintillation(gentype) ); 

     uif_t uifa[4] ;
     uifa[0].i = gentype ; 
     uifa[1].i = parentId ; 
     uifa[2].i = translate(materialId) ;   // raw G4 materialId translated into GBndLib material line for GPU usage 
     uifa[3].i = numPhotons ; 

     uif_t uifb[4] ;
     uifb[0].i = pdgCode ;
     uifb[1].i = scntId ;
     uifb[2].i = 0 ;
     uifb[3].i = 0 ;

     //////////// 6*4 floats for one step ///////////

     float* ss = m_genstep_values ; 

     ss[0*4+0] = uifa[0].f ;
     ss[0*4+1] = uifa[1].f ;
     ss[0*4+2] = uifa[2].f ;
     ss[0*4+3] = uifa[3].f ;

     ss[1*4+0] = x0_x ;
     ss[1*4+1] = x0_y ;
     ss[1*4+2] = x0_z ;
     ss[1*4+3] = t0 ;

     ss[2*4+0] = deltaPosition_x ;
     ss[2*4+1] = deltaPosition_y ;
     ss[2*4+2] = deltaPosition_z ;
     ss[2*4+3] = stepLength ;

     ss[3*4+0] = uifb[0].f ;  // pdgCode
     ss[3*4+1] = pdgCharge ;
     ss[3*4+2] = weight ;
     ss[3*4+3] = meanVelocity ;

     ss[4*4+0] = uifb[1].f ;  // scntId
     ss[4*4+1] = slowerRatio ;
     ss[4*4+2] = slowTimeConstant ;
     ss[4*4+3] = slowerTimeConstant ;

     ss[5*4+0] = scintillationTime ;
     ss[5*4+1] = scintillationIntegralMax ;
     ss[5*4+2] = spare1 ;
     ss[5*4+3] = spare2 ;

     m_genstep->add(ss, m_genstep_itemsize);
}



void CGenstepCollector::collectCerenkovStep
(
            G4int                gentype, 
            G4int                parentId,
            G4int                materialId,
            G4int                numPhotons,
            
            G4double             x0_x,  
            G4double             x0_y,  
            G4double             x0_z,  
            G4double             t0, 

            G4double             deltaPosition_x, 
            G4double             deltaPosition_y, 
            G4double             deltaPosition_z, 
            G4double             stepLength, 

            G4int                pdgCode, 
            G4double             pdgCharge, 
            G4double             weight, 
            G4double             preVelocity, 

            G4double             betaInverse,
            G4double             pmin,
            G4double             pmax,
            G4double             maxCos,

            G4double             maxSin2,
            G4double             meanNumberOfPhotons1,
            G4double             meanNumberOfPhotons2,
            G4double             postVelocity
)
{
     m_cerenkov_count += 1 ;   
     m_gs_photons.push_back(numPhotons); 

     LOG(LEVEL)
          << " gentype " << gentype
          << " gentype " << OpticksGenstep::Gentype(gentype)
          << " pdgCode " << pdgCode
          << " numPhotons " << numPhotons 
          << desc()
          ;

     assert( OpticksGenstep::IsCerenkov(gentype) ); 

     uif_t uifa[4] ;
     uifa[0].i = gentype ; 
     uifa[1].i = parentId ; 
     uifa[2].i = translate(materialId) ; 
     uifa[3].i = numPhotons ; 

     uif_t uifb[4] ;
     uifb[0].i = pdgCode ;
     uifb[1].i = 0 ;
     uifb[2].i = 0 ;
     uifb[3].i = 0 ;

     //////////// 6*4 floats for one step ///////////

     float* cs = m_genstep_values ; 

     cs[0*4+0] = uifa[0].f ;
     cs[0*4+1] = uifa[1].f ;
     cs[0*4+2] = uifa[2].f ;
     cs[0*4+3] = uifa[3].f ;

     cs[1*4+0] = x0_x ;
     cs[1*4+1] = x0_y ;
     cs[1*4+2] = x0_z ;
     cs[1*4+3] = t0 ;

     cs[2*4+0] = deltaPosition_x ;
     cs[2*4+1] = deltaPosition_y ;
     cs[2*4+2] = deltaPosition_z ;
     cs[2*4+3] = stepLength ;

     cs[3*4+0] = uifb[0].f ;  // pdgCode
     cs[3*4+1] = pdgCharge ;
     cs[3*4+2] = weight ;
     cs[3*4+3] = preVelocity ;

     cs[4*4+0] = betaInverse ;  
     cs[4*4+1] = pmin ;
     cs[4*4+2] = pmax ;
     cs[4*4+3] = maxCos ;

     cs[5*4+0] = maxSin2 ;  
     cs[5*4+1] = meanNumberOfPhotons1 ;
     cs[5*4+2] = meanNumberOfPhotons2 ;
     cs[5*4+3] = postVelocity ;

     m_genstep->add(cs, m_genstep_itemsize);
}

/**
CGenstepCollector::collectMachineryStep
-----------------------------------------

Duplicates the gentype as a float into normal genstep itemsize.

**/

void CGenstepCollector::collectMachineryStep(unsigned gentype)
{
     assert( OpticksGenstep::IsMachinery(gentype) ); 

     m_machinery_count += 1 ;  
     LOG(debug) 
           << " machinery_count " << m_machinery_count ;


     float* ms = m_genstep_values ; 

     uif_t uif ; 
     uif.u = gentype ; 
     for(unsigned i=0 ; i < m_genstep_itemsize ; i++) ms[i] = uif.f ;  

     m_genstep->add(ms, m_genstep_itemsize);
}


/**
CGenstepCollector::collectOpticksGenstep
------------------------------------------

Experimental for debugging only. Used by g4ok/G4OKTest 
Currently expects only torchsteps.

**/

void CGenstepCollector::collectOpticksGenstep(const OpticksGenstep* gs)
{
    unsigned num_gensteps = gs->getNumGensteps(); 
    const NPY<float>* src = gs->getGensteps(); 
    float* dst_values = m_genstep_values ; 


    LOG(LEVEL) 
        << " num_gensteps " << num_gensteps 
        << " src.shape " << src->getShapeString()
        << " src_oac.desc " << OpticksActionControl::Desc(src->getActionControl())
        ;

    for(unsigned idx=0 ; idx < num_gensteps ; idx++)
    {
        unsigned gentype = gs->getGencode(idx); 
        LOG(LEVEL) 
            << " idx " << idx
            << " gentype " << gentype
            ;

        if( OpticksGenstep::IsCerenkov(gentype) )
        {
            assert(0); 
            m_cerenkov_count += 1 ;  
        }
        else if( OpticksGenstep::IsScintillation(gentype) )
        {
            assert(0); 
            m_scintillation_count += 1 ;  
        }
        else if( OpticksGenstep::IsTorchLike(gentype) )
        {
            m_torch_count += 1 ;  

            bool emitsource = OpticksGenstep::IsEmitSource(gentype) ; 
            m_torch_emitsource_count += int(emitsource) ;  

            const float* src_values = src->getValuesConst(idx);  
            for(unsigned j=0 ; j < m_genstep_itemsize ; j++) dst_values[j] = src_values[j] ; 
            m_genstep->add(dst_values, m_genstep_itemsize);
        }
        else if( OpticksGenstep::IsMachinery(gentype) )
        {
            assert(0); 
            m_machinery_count += 1 ;  
        }
        else
        {
            assert(0); 
        }
    } 

   
    if(m_torch_emitsource_count > 0 )
    {
        LOG(LEVEL) 
            << " num_gensteps " << num_gensteps
            << " m_torch_count " << m_torch_count
            << " m_torch_emitsource_count " << m_torch_emitsource_count
            ;         

        assert( m_torch_count == m_torch_emitsource_count ); 
        assert( 1 == m_torch_emitsource_count ); 
        assert( num_gensteps == m_torch_emitsource_count );

        m_genstep->setActionControl(src->getActionControl());   
        m_genstep->setAux(src->getAux()); 
        LOG(LEVEL) 
           << " torch_emitsource pass along " 
           << " oac " << OpticksActionControl::Desc(m_genstep->getActionControl()) 
           << " aux " << m_genstep->getAux() 
           ; 
    }


}

 
