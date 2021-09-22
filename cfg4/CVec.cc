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

#include "SDigest.hh"
#include "SDirect.hh"

#include "CVec.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include "PLOG.hh"

CVec::CVec(G4MaterialPropertyVector* vec ) 
   :
   m_vec(vec)
{
}

G4MaterialPropertyVector* CVec::getVec()
{
    return m_vec ; 
}


CVec* CVec::MakeDummy(size_t n )
{
    double e[n] ; 
    double v[n] ; 
    for(unsigned i=0 ; i < n ; i++ )
    {
        e[i] = double(i)*100 + 0.1 ;  
        v[i] = double(i)*1000 + 0.2 ;  
    }   

    G4MaterialPropertyVector* _vec = new G4MaterialPropertyVector(e, v, n ); 
    CVec* vec = new CVec(_vec) ; 
    return vec ; 
 }


std::string CVec::digest()
{
    return Digest(this) ;   
}

std::string CVec::Digest(CVec* vec)  
{
    return Digest(vec ? vec->getVec() : NULL ); 
}

std::string CVec::Digest(G4MaterialPropertyVector* vec)  // see G4PhysicsOrderedFreeVectorTest
{      
    if(!vec) return "" ; 

    std::ofstream fp("/dev/null", std::ios::out);
    std::stringstream ss ;          
    stream_redirect rdir(ss,fp); // stream_redirect such that writes to the file instead go to the stringstream 
       
    vec->Store(fp, false );
       
    std::string s = ss.str();  // std::string can hold \0  (ie that is not interpreted as null terminator) so they can hold any binary data 

    SDigest dig ;
    dig.update( const_cast<char*>(s.data()), s.size() );
       
    return dig.finalize();
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


