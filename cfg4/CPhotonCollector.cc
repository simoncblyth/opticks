#include "NPY.hpp"
#include "CPhotonCollector.hh"
#include "PLOG.hh"

CPhotonCollector* CPhotonCollector::INSTANCE = NULL ;

CPhotonCollector* CPhotonCollector::Instance()
{
   assert(INSTANCE && "CPhotonCollector has not been instanciated");
   return INSTANCE ;
}


CPhotonCollector::CPhotonCollector()    
    :
    m_photon(NPY<float>::make(0,4,4)),
    m_photon_itemsize(m_photon->getNumValues(1)),
    m_photon_values(new float[m_photon_itemsize]),
    m_photon_count(0)
{
    assert( m_photon_itemsize == 4*4 );
    INSTANCE = this ; 
}


NPY<float>*  CPhotonCollector::getPhoton() const 
{
    return m_photon ; 
}

void CPhotonCollector::save(const char* path) const 
{
    m_photon->save(path) ; 
}

std::string CPhotonCollector::description() const
{
    std::stringstream ss ; 
    ss << " CPhotonCollector "
       << " photon_count " << m_photon_count
       ;
    return ss.str();
}

void CPhotonCollector::Summary(const char* msg) const 
{ 
    LOG(info) << msg 
              << description()
              ;
}


void CPhotonCollector::collectPhoton(

               G4double  pos_x,
               G4double  pos_y,
               G4double  pos_z,
               G4double  time,

               G4double  dir_x,
               G4double  dir_y,
               G4double  dir_z,
               G4double  weight,

               G4double  pol_x,
               G4double  pol_y,
               G4double  pol_z,
               G4double  wavelength,

               int flags_x,
               int flags_y,
               int flags_z,
               int flags_w
          )
{
     float* pr = m_photon_values ; 
     
     pr[0*4+0] = pos_x ;
     pr[0*4+1] = pos_y ;
     pr[0*4+2] = pos_z ;
     pr[0*4+3] = time ;

     pr[1*4+0] = dir_x ;
     pr[1*4+1] = dir_y ;
     pr[1*4+2] = dir_z ;
     pr[1*4+3] = weight  ;

     pr[2*4+0] = pol_x ;
     pr[2*4+1] = pol_y ;
     pr[2*4+2] = pol_z ;
     pr[2*4+3] = wavelength  ;

     uif_t flags[4] ;
     flags[0].i = flags_x ;   
     flags[1].i = flags_y ;   
     flags[2].i = flags_z ;   
     flags[3].i = flags_w ;   

     pr[3*4+0] = flags[0].f ;
     pr[3*4+1] = flags[1].f ;
     pr[3*4+2] = flags[2].f ;
     pr[3*4+3] = flags[3].f  ;

     m_photon->add(pr, m_photon_itemsize);
}



