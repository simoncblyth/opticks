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


Example 0 

       u = [0 0 1]
       up = 0. 
       u.z < 0. ?  no  

      -> dx,dy,dz  vector is unchanged because original frame Uz already in Z-direction


Example 1 

       u = [0,0,-1]   
       up = 0.   axial z 
       u.z < 0.  yes

                | -1  0   0  |  
         d ->   |  0  1   0  | d
                |  0  0  -1  |  

   Flipping X and Z and leaving Y 

        Z 
        |            
        +--X        -X---+
       /                /|
      Y               Y  -Z



Example 2 

        u = [0,1,1]     ## it should be normalized


                 |   0 -1  0  |   
                 |   1  0  1  | 
                 |  -1  0  1  |


          x' = -y
          y' = 


                |  u.x * u.z / up   -u.y / up    u.x  |        
        d  =    |  u.y * u.z / up   +u.x / up    u.y  |      p
                |   -up               0.         u.z  |      
    
                                   

The columns of a rotation matrix are orthogonal unit vectors. A rotation matrix
may transform any set of vectors, so we can consider transforming the three
unit vectors along the x, y and z axes, which by definition are orthogonal to
each other.

Conside normalization of the columns

* 3rd column: by requirement that u is normalized

* 1st column also follows:

      ( u.x*u.x + u.y*u.y ) u.z*u.z + up*up =  ux^2 + uy^2 + uz^2 = 1 
       --------------------
            up*up
                                   
* 2nd column also follows from up being xy radius: 
  
           u.y*u.y + u.x*u.x 
          -------------------- = 1 
                  up*up 

Consider dot products between the columns, 

* first and third::

      (ux*ux + uy*uy) * uz - up*uz          up*up * uz
       -------------------            =    ------------  -  up*uz = 0 
              up                                up

* second and third: clearly zero from x y symmetry 
* first and second : clearly zero from x y symmetry 
   

The above observations of the normalization and orthogonality 
show that it is indeed a rotation matrix. 


https://root.cern.ch/phpBB3/viewtopic.php?t=7421

 TVector3 direction = v.Unit() 
  v1.RotateUz(direction); // direction must be TVector3 of unit length

transforms v1 from the rotated frame 
   (z' parallel to direction, 
    x' in the theta plane and 
    y' in the xy plane as well as perpendicular to the theta plane) 
 to the (x,y,z) frame.


https://proj-clhep.web.cern.ch/proj-clhep/manual/UserGuide/VectorDefs/node49.html

This rotates the reference frame such that the original Z-axis will lie in the
direction of $\hat{u}$. Many rotations would accomplish this; the one selected
uses $u$ as its third column and is given by: 


g4-cls ThreeVector::

    199   Hep3Vector & rotateUz(const Hep3Vector&);
    200   // Rotates reference frame from Uz to newUz (unit vector) (Geant4).

    038 Hep3Vector & Hep3Vector::rotateUz(const Hep3Vector& NewUzVector) {
     39   // NewUzVector must be normalized !
     40 
     41   double u1 = NewUzVector.x();
     42   double u2 = NewUzVector.y();
     43   double u3 = NewUzVector.z();
     44   double up = u1*u1 + u2*u2;
     45     
     46   if (up>0) {
     47       up = std::sqrt(up);
     //       radius in xy plane 
     48       double px = dx,  py = dy,  pz = dz;
     49       dx = (u1*u3*px - u2*py)/up + u1*pz;
     50       dy = (u2*u3*px + u1*py)/up + u2*pz;
     51       dz =    -up*px +             u3*pz;




                |  u.x * u.z / up   -u.y / up    u.x  |        
        d  =    |  u.y * u.z / up   +u.x / up    u.y  |      p
                |   -up               0.         u.z  |      
    



     52     }
     53   else if (u3 < 0.) { dx = -dx; dz = -dz; }      // phi=0  teta=pi
     54   else {};
     55   return *this;
     56 }

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



