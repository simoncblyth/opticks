#include "NPY.hpp"
#include "CPrimaryCollector.hh"
#include "PLOG.hh"

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "G4Event.hh"
#include "G4PrimaryVertex.hh"
#include "G4PrimaryParticle.hh"
#include "G4ParticleDefinition.hh"


CPrimaryCollector* CPrimaryCollector::INSTANCE = NULL ;

CPrimaryCollector* CPrimaryCollector::Instance()
{
   assert(INSTANCE && "CPrimaryCollector has not been instanciated");
   return INSTANCE ;
}


CPrimaryCollector::CPrimaryCollector()    
    :
    m_primary(NPY<float>::make(0,4,4)),
    m_primary_itemsize(m_primary->getNumValues(1)),
    m_primary_values(new float[m_primary_itemsize]),
    m_primary_count(0)
{
    assert( m_primary_itemsize == 4*4 );
    INSTANCE = this ; 
}


NPY<float>*  CPrimaryCollector::getPrimary() const 
{
    return m_primary ; 
}

void CPrimaryCollector::save(const char* path) const 
{
    m_primary->save(path) ; 
}



std::string CPrimaryCollector::description() const
{
    std::stringstream ss ; 
    ss << " CPrimaryCollector "
       << " primary_count " << m_primary_count
       ;
    return ss.str();
}

void CPrimaryCollector::Summary(const char* msg) const 
{ 
    LOG(info) << msg 
              << description()
              ;
}

void CPrimaryCollector::collectPrimaries(const G4Event* event)
{
    G4int num_v = event->GetNumberOfPrimaryVertex() ;
    LOG(info) << " num_v " << num_v ; 
    for(G4int v=0 ; v < num_v ; v++) collectPrimaryVertex(v, event); 
}

void CPrimaryCollector::collectPrimaryVertex(G4int vertex_index, const G4Event* event)
{
    const G4PrimaryVertex* vtx = event->GetPrimaryVertex(vertex_index) ;
    collectPrimaryVertex(vtx,  vertex_index);
}

void CPrimaryCollector::collectPrimaryVertex(const G4PrimaryVertex* vtx, G4int vertex_index)
{
    G4int num_p = vtx->GetNumberOfParticle() ;
    LOG(info) << " vtx " << vtx << " num_p " << num_p ;    
    for(G4int p=0 ; p < num_p ; p++) collectPrimaryParticle(vertex_index, p, vtx ) ; 
}

void CPrimaryCollector::collectPrimaryParticle(G4int vertex_index, G4int primary_index, const G4PrimaryVertex* vtx)
{
    G4double time = vtx->GetT0() ;
    G4PrimaryParticle* pp = vtx->GetPrimary(primary_index); 

    const G4ParticleDefinition* pd = pp->GetParticleDefinition();  
    G4int pdgcode = pp->GetPDGcode() ; 
    LOG(info) 
        << " pp " << pp  
        << " pdgcode " << pdgcode
        << " pd " << pd->GetParticleName() 
        ;   

    G4ThreeVector pos = vtx->GetPosition() ;

    const G4ThreeVector& dir = pp->GetMomentumDirection()  ; 
    G4ThreeVector pol = pp->GetPolarization() ;
  
    G4double kineticEnergy = pp->GetKineticEnergy()  ; 
    //G4double totalEnergy = pp->GetTotalEnergy() ; // includes mass 
    G4double weight = pp->GetWeight() ; 

    collectPrimary(

           pos.x()/mm,
           pos.y()/mm,
           pos.z()/mm,
           time/ns,

           dir.x(),
           dir.y(),
           dir.z(),
           weight,

           pol.x(),
           pol.y(),
           pol.z(),
           kineticEnergy/MeV,

           0u,
           vertex_index,
           primary_index,
           pdgcode 
               
         ); 
}


void CPrimaryCollector::collectPrimary(
               G4double  x0,
               G4double  y0,
               G4double  z0,
               G4double  t0,

               G4double  dir_x,
               G4double  dir_y,
               G4double  dir_z,
               G4double  weight,

               G4double  pol_x,
               G4double  pol_y,
               G4double  pol_z,
               G4double  kineticEnergy,

               int spare,
               int vertex_index,
               int primary_index,
               int pdgcode
          )
{
     float* pr = m_primary_values ; 
     
     pr[0*4+0] = x0 ;
     pr[0*4+1] = y0 ;
     pr[0*4+2] = z0 ;
     pr[0*4+3] = t0 ;

     pr[1*4+0] = dir_x ;
     pr[1*4+1] = dir_y ;
     pr[1*4+2] = dir_z ;
     pr[1*4+3] = weight  ;

     pr[2*4+0] = pol_x ;
     pr[2*4+1] = pol_y ;
     pr[2*4+2] = pol_z ;
     pr[2*4+3] = kineticEnergy  ;

     uif_t flags[4] ;
     flags[0].i = spare ;   
     flags[1].i = vertex_index ;   
     flags[2].i = primary_index ;   
     flags[3].i = pdgcode ;   

     pr[3*4+0] = flags[0].f ;
     pr[3*4+1] = flags[1].f ;
     pr[3*4+2] = flags[2].f ;
     pr[3*4+3] = flags[3].f  ;

     m_primary->add(pr, m_primary_itemsize);
}





