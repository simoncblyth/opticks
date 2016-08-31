#include <sstream>

#include "NPY.hpp"
#include "OpticksG4Collector.hh"
#include "PLOG.hh"

OpticksG4Collector* OpticksG4Collector::INSTANCE = NULL ;

OpticksG4Collector* OpticksG4Collector::Instance()
{
   if(!INSTANCE) INSTANCE = new OpticksG4Collector ;
   return INSTANCE ;
}

OpticksG4Collector::OpticksG4Collector()  
    :
    m_onestep(NPY<float>::make(1,6,4)),
    m_values(NULL),
    m_genstep(NPY<float>::make(0,6,4)),
    m_scintillation_count(0),
    m_cerenkov_count(0)
{
    m_onestep->zero();
    m_values = m_onestep->getValues();
}

NPY<float>*  OpticksG4Collector::getGensteps()
{
    consistencyCheck() ;
    return m_genstep ; 
}

void OpticksG4Collector::consistencyCheck()
{
     unsigned numItems = m_genstep->getNumItems();
     bool consistent = numItems == m_scintillation_count + m_cerenkov_count ;
     if(!consistent)
         LOG(fatal) << "OpticksG4Collector::consistencyCheck FAIL " 
                    << description()
                    ;
     assert(consistent);
}

std::string OpticksG4Collector::description()
{
    std::stringstream ss ; 
    ss << " OpticksG4Collector "
       << " numItems " << m_genstep->getNumItems() 
       << " scintillation_count " << m_scintillation_count
       << " cerenkov_count " << m_cerenkov_count
       << " step_count " << m_scintillation_count + m_cerenkov_count
       ;
    return ss.str();
}

void OpticksG4Collector::Summary(const char* msg)
{ 
    LOG(info) << msg 
              << description()
              ;
}


void OpticksG4Collector::collectScintillationStep
(
            G4int                id, 
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
     LOG(info) << " scintillation_count " << m_scintillation_count ;

     uif_t uifa[4] ;
     uifa[0].i = id == 0 ? m_scintillation_count : id  ;   // use the 1-based index when id zero 
     uifa[1].i = parentId ; 
     uifa[2].i = materialId ;   // raw G4 materialId, will need to be translated into GBndLib material line for GPU usage 
     uifa[3].i = numPhotons ; 

     uif_t uifb[4] ;
     uifb[0].i = pdgCode ;
     uifb[1].i = scntId ;
     uifb[2].i = 0 ;
     uifb[3].i = 0 ;

     //////////// 6*4 floats for one step ///////////

     float* ss = m_values ; 

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

     m_genstep->add(m_onestep);
}



void OpticksG4Collector::collectCerenkovStep
(
            G4int                id, 
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

            G4double             betaInverse,
            G4double             pmin,
            G4double             pmax,
            G4double             maxCos,

            G4double             maxSin2,
            G4double             meanNumberOfPhotons1,
            G4double             meanNumberOfPhotons2,
            G4double             spare2
)
{
     m_cerenkov_count += 1 ;   // 1-based index
     LOG(info) << " cerenkov_count " << m_cerenkov_count ;

     uif_t uifa[4] ;
     uifa[0].i = id == 0 ? -m_cerenkov_count : id  ;   // use the negated 1-based index when id zero 
     uifa[1].i = parentId ; 
     uifa[2].i = materialId ; 
     uifa[3].i = numPhotons ; 

     uif_t uifb[4] ;
     uifb[0].i = pdgCode ;
     uifb[1].i = 0 ;
     uifb[2].i = 0 ;
     uifb[3].i = 0 ;

     //////////// 6*4 floats for one step ///////////

     float* cs = m_values ; 

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
     cs[3*4+3] = meanVelocity ;

     cs[4*4+0] = betaInverse ;  
     cs[4*4+1] = pmin ;
     cs[4*4+2] = pmax ;
     cs[4*4+3] = maxCos ;

     cs[5*4+0] = maxSin2 ;  
     cs[5*4+1] = meanNumberOfPhotons1 ;
     cs[5*4+2] = meanNumberOfPhotons2 ;
     cs[5*4+3] = spare2 ;

     m_genstep->add(m_onestep);
}

 
