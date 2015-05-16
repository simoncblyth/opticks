#pragma once

/*

 // geant4.10.00.p01/source/externals/clhep/src/ThreeVector.cc

 72 Hep3Vector & Hep3Vector::rotateUz(const Hep3Vector& NewUzVector) {
 73   // NewUzVector must be normalized !
 74 
 75   double u1 = NewUzVector.x();
 76   double u2 = NewUzVector.y();
 77   double u3 = NewUzVector.z();
 78   double up = u1*u1 + u2*u2;
 79 
 80   if (up>0) {
 81       up = std::sqrt(up);
 82       double px = dx,  py = dy,  pz = dz;
 83       dx = (u1*u3*px - u2*py)/up + u1*pz;
 84       dy = (u2*u3*px + u1*py)/up + u2*pz;
 85       dz =    -up*px +             u3*pz;
 86     }
 87   else if (u3 < 0.) { dx = -dx; dz = -dz; }      // phi=0  teta=pi
 88   else {};
 89   return *this;
 90 }

*/

__device__ void rotateUz(float3& d, float3& u )
{
    float up = u.x*u.x + u.y*u.y ;
    if (up>0.f) 
    {
       up = sqrt(up);

       float px = d.x ;
       float py = d.y ;
       float pz = d.z ;

       d.x = (u.x*u.z*px - u.y*py)/up + u.x*pz;
       d.y = (u.y*u.z*px + u.x*py)/up + u.y*pz;
       d.z =    -up*px +                u.z*pz;
   }
   else if (u.z < 0.f ) 
   { 
       d.x = -d.x; 
       d.z = -d.z; 
   }                  // phi=0  teta=pi
   else {};
   
}



