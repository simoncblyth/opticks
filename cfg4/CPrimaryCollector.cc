#include "NPY.hpp"
#include "CPrimaryCollector.hh"
#include "PLOG.hh"


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
               G4double  wavelength,

               unsigned flags_x,
               unsigned flags_y,
               unsigned flags_z,
               unsigned flags_w
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
     pr[2*4+3] = wavelength  ;

     uif_t flags[4] ;
     flags[0].u = flags_x ;   
     flags[1].u = flags_y ;   
     flags[2].u = flags_z ;   
     flags[3].u = flags_w ;   

     pr[3*4+0] = flags[0].f ;
     pr[3*4+1] = flags[1].f ;
     pr[3*4+2] = flags[2].f ;
     pr[3*4+3] = flags[3].f  ;

     m_primary->add(pr, m_primary_itemsize);
}



