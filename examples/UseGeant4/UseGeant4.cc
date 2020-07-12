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
#include "G4Version.hh"
#include "G4Polycone.hh"
#include "G4AffineTransform.hh"
#include "G4ThreeVector.hh"

using CLHEP::deg ; 


struct Test 
{
   static void dump_version(); 
   static void make_polycone_0(); 
   static void make_polycone_1(); 
   static void make_transform(); 
};



void Test::dump_version()
{
    std::cout << "G4VERSION_NUMBER " << G4VERSION_NUMBER << std::endl ; 
    std::cout << "G4VERSION_TAG    " << G4VERSION_TAG << std::endl ; 
    std::cout << "G4Version        " << G4Version << std::endl ; 
    std::cout << "G4Date           " << G4Date << std::endl ; 
}


void Test::make_polycone_0()
{
     G4double phiStart = 0.00*deg ; 
     G4double phiTotal = 360.00*deg ; 
     G4int numRZ = 2 ; 
     G4double r[] = {50.000999999999998, 75.82777395122217} ; 
     G4double z[] = {-19.710672039327765, 19.710672039327765} ; 
    
     G4Polycone* pc = new G4Polycone("name", phiStart, phiTotal, numRZ, r, z ); 
     G4cout << *pc << std::endl ; 
}

void Test::make_polycone_1()
{
     G4double phiStart = 0.00*deg ; 
     G4double phiTotal = 360.00*deg ; 
     G4int numZPlanes = 2 ; 

     G4double zPlane[] = {-19.710672039327765, 19.710672039327765} ; 
     G4double rInner[] = {0.0, 0.0} ; 
     G4double rOuter[] = {50.000999999999998, 75.82777395122217} ; 
    
     G4Polycone* pc = new G4Polycone("name", phiStart, phiTotal, numZPlanes, zPlane, rInner, rOuter ); 
     G4cout << *pc << std::endl ; 
}


G4AffineTransform X4AffineTransform__FromTransform(const G4Transform3D& T ) 
{
    // duplicate from CFG4.CMath

    G4ThreeVector colX(T.xx(), T.xy(), T.xz());
    G4ThreeVector colY(T.yx(), T.yy(), T.yz());
    G4ThreeVector colZ(T.zx(), T.zy(), T.zz());

    G4RotationMatrix rot(colX,colY,colZ) ;
    G4ThreeVector tlate(T.dx(), T.dy(), T.dz());

    return G4AffineTransform( rot, tlate) ; 
}

G4RotationMatrix G4GDMLReadDefine__GetRotationMatrix(const G4ThreeVector& angles)
{
   G4RotationMatrix rot;

   rot.rotateX(angles.x());
   rot.rotateY(angles.y());
   rot.rotateZ(angles.z());
   rot.rectify();  // Rectify matrix from possible roundoff errors

   return rot;
}



void Test::make_transform()
{

     // numbers grabbed from from debug session
/*
reakpoint 8, junoSD_PMT_v2::ProcessHits (this=0x34d0f10, step=0x252ccf0) at ../src/junoSD_PMT_v2.cc:277
277	    double qe = 1;
(gdb) p global_pos
$6 = (const G4ThreeVector &) @0x252ce20: {dx = -7250.5045525891683, dy = 17122.963751776308, dz = -5263.5969960140847, static tolerance = 2.22045e-14}
(gdb) p local_pos
$7 = {dx = -112.67072395684227, dy = 165.92175413608675, dz = 109.63878699927591, static tolerance = 2.22045e-14}
(gdb) p trans
$8 = (const G4AffineTransform &) @0x252ff58: {rxx = -0.10182051317974285, rxy = -0.92429043017162327, rxz = 0.36785837463481702, ryx = 0.24656591428433955, ryy = -0.38168992741904467, 
  ryz = -0.89079630063217707, rzx = 0.96376233222145669, rzy = 0, rzz = 0.26676237926487772, tx = -0.0035142754759363015, ty = 0.012573876562782971, tz = 19434.000031086449}
(gdb) p track->GetVolume()->GetCopyNo()

*/

     double rxx = -0.10182051317974285 ;
     double rxy = -0.92429043017162327 ;
     double rxz =  0.36785837463481702 ; 

     double ryx =  0.24656591428433955 ;
     double ryy = -0.38168992741904467 ; 
     double ryz = -0.89079630063217707 ;

     double rzx =  0.96376233222145669 ;
     double rzy = 0 ;
     double rzz = 0.26676237926487772 ;  

     /*
     G4ThreeVector colX(rxx, rxy, rxz);
     G4ThreeVector colY(ryx, ryy, ryz);
     G4ThreeVector colZ(rzx, rzy, rzz);
     */

     G4ThreeVector colX(rxx, ryx, rzx);
     G4ThreeVector colY(rxy, ryy, rzy);
     G4ThreeVector colZ(rxz, ryz, rzz);

     G4RotationMatrix dbg_rot(colX,colY,colZ) ;  // from the G4AffineTransform debugging

     G4cout << "dbg_rot " << dbg_rot << G4endl ; 

     double tx = -0.0035142754759363015 ; 
     double ty =  0.012573876562782971 ;  
     double tz =  19434.000031086449 ;
     G4ThreeVector    dbg_tla(tx, ty, tz);

     G4AffineTransform dbg_affine_trans(dbg_rot,dbg_tla); 

     G4cout << "dbg_affine_trans " << dbg_affine_trans << G4endl ; 
     

     G4ThreeVector global[3] ; 
     global[0] = { 0, 0, 0 } ; 
     global[1] = { -7250.5045525891683,17122.963751776308,-5263.5969960140847 } ; // from photon hit position from debug session
     global[2] = { -7148.9484,         17311.741,         -5184.2567          } ; // from GDML physvol for the corres pmtid /copynumber
/*
     71423       <physvol copynumber="11336" name="pLPMT_Hamamatsu_R128600x353fc90">
     71424         <volumeref ref="HamamatsuR12860lMaskVirtual0x3290b70"/>
     71425         <position name="pLPMT_Hamamatsu_R128600x353fc90_pos" unit="mm" x="-7148.9484" y="17311.741" z="-5184.2567"/>
     71426         <rotation name="pLPMT_Hamamatsu_R128600x353fc90_rot" unit="deg" x="-73.3288783033161" y="-21.5835981926051" z="-96.2863976680901"/>
     71427       </physvol>

*/

    std::cout << "local = dbg_affine_trans.TransformPoint(global[i]) " << std::endl ; 

    G4ThreeVector local ; 
    for(int i=0 ; i < 3 ; i++)
    {
        local = dbg_affine_trans.TransformPoint(global[i]);  
        std::cout 
            << " gx " << global[i].x()
            << " gy " << global[i].y()
            << " gz " << global[i].z()
            << " lx " << local.x()
            << " ly " << local.y()
            << " lz " << local.z()
            << std::endl 
            ;
    }  


    // g4-cls G4GDMLReadStructure

    G4ThreeVector rotation(-73.3288783033161*deg, -21.5835981926051*deg, -96.2863976680901*deg); 

    G4double s = 1. ; 
    G4ThreeVector position(-7148.9484*s, 17311.741*s, -5184.2567*s);
 
    G4RotationMatrix gdml_rot = G4GDMLReadDefine__GetRotationMatrix(rotation).inverse() ;  
    //G4RotationMatrix gdml_rot = G4GDMLReadDefine__GetRotationMatrix(rotation) ;  
    G4cout << "gdml_rot " << gdml_rot << G4endl ; 

    G4Transform3D t3(gdml_rot,position);   // g4-cls G4GDMLReadStructure
    G4AffineTransform trans2 = X4AffineTransform__FromTransform(t3);  



     G4cout << "trans2 " << trans2 << G4endl ; 
    std::cout << "local = trans2.TransformPoint(global[i]) " << std::endl ; 

    for(int i=0 ; i < 3 ; i++)
    {
        local = trans2.TransformPoint(global[i]);  
        std::cout 
            << " gx " << global[i].x()
            << " gy " << global[i].y()
            << " gz " << global[i].z()
            << " lx " << local.x()
            << " ly " << local.y()
            << " lz " << local.z()
            << std::endl 
            ;
     }
}

int main()
{
    Test::dump_version(); 
    //Test::make_polycone_0(); 
    //Test::make_polycone_1(); 

    Test::make_transform(); 

    return 0 ; 
}
