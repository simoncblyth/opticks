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


#include <cassert>
#include "CFG4_BODY.hh"

#include "PLOG.hh"
#include "CFG4_LOG.hh"

#include "NPY.hpp"

#include "globals.hh"
#include "Randomize.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"

#include "GVector.hh"
#include "Format.hh"

using CLHEP::twopi ; 



guint4 rayleigh_scatter(const G4ThreeVector& OldMomentumDirection, 
                     const G4ThreeVector& OldPolarization, 
                           G4ThreeVector& NewMomentumDirection,
                           G4ThreeVector& NewPolarization)
{ 
    G4double cosTheta;
    G4double rand, constant;
    G4double CosTheta, SinTheta, SinPhi, CosPhi, unit_x, unit_y, unit_z;
    G4ThreeVector UniformSphere ;

    guint4 rc ;  
    rc.x = 0 ; 
    rc.y = 0 ; 
    rc.z = 0 ; 
    rc.w = 0 ; 

    do
    {
        CosTheta = G4UniformRand() ; 
        SinTheta = std::sqrt(1.-CosTheta*CosTheta);
        if (G4UniformRand() < 0.5) CosTheta = -CosTheta;

        rand = twopi*G4UniformRand();
        SinPhi = std::sin(rand);
        CosPhi = std::cos(rand);

        unit_x = SinTheta * CosPhi;
        unit_y = SinTheta * SinPhi;
        unit_z = CosTheta; 


        UniformSphere.set(unit_x,unit_y,unit_z);
         
        NewMomentumDirection = UniformSphere ;

        // Rotate the new momentum direction into global reference system
        NewMomentumDirection.rotateUz(OldMomentumDirection);
        NewMomentumDirection = NewMomentumDirection.unit();

        // calculate the new polarization direction
        // The new polarization needs to be in the same plane as the new
        // momentum direction and the old polarization direction
        constant = -NewMomentumDirection.dot(OldPolarization);

        NewPolarization = OldPolarization + constant*NewMomentumDirection;
        NewPolarization = NewPolarization.unit(); 


        // There is a corner case, where the Newmomentum direction
        // is the same as oldpolariztion direction:
        // random generate the azimuthal angle w.r.t. Newmomentum direction
        if (NewPolarization.mag() == 0.) 
        {
            rc.y++ ; 
            rand = G4UniformRand()*twopi;
            NewPolarization.set(std::cos(rand),std::sin(rand),0.);
            NewPolarization.rotateUz(NewMomentumDirection);
        } 
        else 
        {
            // There are two directions which are perpendicular
            // to the new momentum direction
            if (G4UniformRand() < 0.5) NewPolarization = -NewPolarization;
        }

        // simulate according to the distribution cos^2(theta)
        cosTheta = NewPolarization.dot(OldPolarization);
        rc.x++ ; 
    }
    while (cosTheta*cosTheta < G4UniformRand());
    //while (std::pow(cosTheta,2) < G4UniformRand());
    // Loop checking, 13-Aug-2015, Peter Gumplinger

    return rc ; 
}




int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    CFG4_LOG_ ;

    enum { OLDMOM, OLDPOL, NEWMOM, NEWPOL } ;

    G4ThreeVector OldMomentumDirection(1.f,0.f,0.f) ;
    G4ThreeVector OldPolarization(0.f,1.f,0.f) ;
    OldMomentumDirection = OldMomentumDirection.unit();  
    OldPolarization= OldPolarization.unit();  

    G4ThreeVector NewMomentumDirection;
    G4ThreeVector NewPolarization ;

    unsigned ngen = 1000000 ;  
    NPY<float>* buf = NPY<float>::make(ngen, 4, 4 );
    buf->zero(); 


    guint4 tot(0,0,0,0) ; 

    for(unsigned i=0 ; i < ngen ; i++)
    {
        guint4 rc = rayleigh_scatter(OldMomentumDirection, OldPolarization, NewMomentumDirection, NewPolarization );
       
        tot.x += rc.x ; 
        tot.y += rc.y ; 
        tot.z += rc.z ; 
        tot.w += rc.w ; 


        if(i%1000 == 0) LOG(info) 
            << " i " << std::setw(6) << i
            << " count " << std::setw(6) << rc.x 
            << " corner " << std::setw(6) << rc.y
            << Format(NewMomentumDirection, "DIR",7)
            << Format(NewPolarization, "POL",7)
            << " rc " << rc.description()
            ;

        buf->setQuad(i, OLDMOM, 0, OldMomentumDirection.x(), OldMomentumDirection.y(), OldMomentumDirection.z(), 0.f );
        buf->setQuad(i, OLDPOL, 0, OldPolarization.x(),      OldPolarization.y(),      OldPolarization.z(),      0.f );
        buf->setQuad(i, NEWMOM, 0, NewMomentumDirection.x(), NewMomentumDirection.y(), NewMomentumDirection.z(), 0.f );
        buf->setQuad(i, NEWPOL, 0, NewPolarization.x(),      NewPolarization.y(),      NewPolarization.z(),      0.f );
    } 

    LOG(info) << " tots " << tot.description() ; 


    buf->save("$TMP/RayleighTest/cfg4.npy");
    return 0 ;
}

