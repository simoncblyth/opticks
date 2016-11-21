#include "CVec.hh"

#include "G4PhysicsOrderedFreeVector.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include "PLOG.hh"

CVec::CVec(G4PhysicsOrderedFreeVector* vec ) 
   :
   m_vec(vec)
{
}

float CVec::getValue(float fnm)
{
   G4double wavelength = G4double(fnm)*CLHEP::nm ; 
   G4double photonMomentum = CLHEP::h_Planck*CLHEP::c_light/wavelength ;
   G4double value = m_vec->Value( photonMomentum );
   return float(value) ;  
}

void CVec::dump(const char* msg, float lo, float hi, float step)
{
   LOG(info) << msg ; 

   float wl = lo ; 
   while( wl <= hi )
   {
       float val = getValue(wl);
       std::cout << std::setw(10) << wl << std::setw(20) << val << std::endl   ;
       wl += step ; 
   } 
}


