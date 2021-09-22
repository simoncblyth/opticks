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


#include "G4MaterialPropertyVector.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include "GDomain.hh"
#include "GProperty.hh"

#include "X4PhysicsVector.hh"
#include "OPTICKS_LOG.hh"


struct X4PhysicsVectorTest 
{
    const char* dir ; 
    size_t    veclen ; 
    G4double* energies ; 
    G4double* values   ;
    G4MaterialPropertyVector*  pof ; 

    X4PhysicsVectorTest();

    void visibleSpectrum();
    void opNovice();

    void test_units(); 
    void test_digest(); 
    void test_convert(); 
    void test_g4interpolate(); 


    G4MaterialPropertyVector* make_pof()
    {
        return new G4MaterialPropertyVector(energies, values, veclen) ; 
    } 
};


X4PhysicsVectorTest::X4PhysicsVectorTest()
    :
    dir("$TMP/X4PhysicsVectorTest"),
    veclen(0),
    energies(nullptr),
    values(nullptr),
    pof(nullptr)
{
    //visibleSpectrum();
    opNovice();
    pof = make_pof(); 
}



void X4PhysicsVectorTest::visibleSpectrum()
{
/*
https://en.wikipedia.org/wiki/Visible_spectrum

Violet	380–450 nm	668–789 THz	2.75–3.26 eV
Blue	450–495 nm	606–668 THz	2.50–2.75 eV
Green	495–570 nm	526–606 THz	2.17–2.50 eV
Yellow	570–590 nm	508–526 THz	2.10–2.17 eV
Orange	590–620 nm	484–508 THz	2.00–2.10 eV
Red	    620–750 nm	400–484 THz	1.65–2.00 eV

*/

    veclen = 7 ; 

    energies = new G4double[veclen] ;

    // multiplying by eV (1e-6) gives values in G4 standard energy units (MeV)

    energies[0] = 1.65*eV ;  // red-
    energies[1] = 2.00*eV ;  // orange- 
    energies[2] = 2.10*eV ;  // yellow- 
    energies[3] = 2.17*eV ;  // green-
    energies[4] = 2.50*eV ;  // blue 
    energies[5] = 2.75*eV ;  // violet-
    energies[6] = 3.26*eV ; 
   
    values   = new G4double[veclen] ;   
    values[0] = 0 ; 
    values[1] = 1 ; 
    values[2] = 2 ; 
    values[3] = 3 ; 
    values[4] = 4 ; 
    values[5] = 5 ; 
    values[6] = 6 ; 
}

void X4PhysicsVectorTest::opNovice()
{
    G4double photonEnergy[] =
            { 2.034*eV, 2.068*eV, 2.103*eV, 2.139*eV,
              2.177*eV, 2.216*eV, 2.256*eV, 2.298*eV,
              2.341*eV, 2.386*eV, 2.433*eV, 2.481*eV,
              2.532*eV, 2.585*eV, 2.640*eV, 2.697*eV,
              2.757*eV, 2.820*eV, 2.885*eV, 2.954*eV,
              3.026*eV, 3.102*eV, 3.181*eV, 3.265*eV,
              3.353*eV, 3.446*eV, 3.545*eV, 3.649*eV,
              3.760*eV, 3.877*eV, 4.002*eV, 4.136*eV };


    veclen = sizeof(photonEnergy)/sizeof(G4double); 

    energies = new G4double[veclen] ;
    for(unsigned i=0 ; i < veclen ; i++) energies[i] = photonEnergy[i] ; 

    values   = new G4double[veclen] ;   
    for(unsigned i=0 ; i < veclen ; i++) values[i] = i ; 
}


void X4PhysicsVectorTest::test_units()
{
    double en_eV = 1.65 ; 
    double en = 1.65*eV ; 
    double hc = h_Planck*c_light/(nm*eV) ;   // in nm.eV  ~1239.84
    double wl_nm = hc/en_eV ;

    LOG(info) << " en " << en 
              << " eV " << eV
              << " h_Planck (4.13567e-12) " << h_Planck
              << " c_light (299.792) mm/ns " << c_light
              << " h_Planck*c_light " << h_Planck*c_light
              << " h_Planck*c_light/(nm*eV) ( 1239.84 ) " << (h_Planck*c_light)/(nm*eV) 
              << " wl_nm " << wl_nm 
              ;
}

void X4PhysicsVectorTest::test_convert()
{
    bool nm_domain = true ; 
 
    GProperty<double>* prop = X4PhysicsVector<double>::Convert(pof, nm_domain ) ; 

    prop->SummaryV("prop", 50);

    prop->save(dir, "convert.npy");
}

void X4PhysicsVectorTest::test_g4interpolate()
{
    GDomain<double>* dom = GDomain<double>::GetDefaultDomain() ; 

    GProperty<double>* prop = X4PhysicsVector<double>::Interpolate(pof, dom ) ; 

    prop->SummaryV("prop", 50);

    prop->save(dir, "g4interpolate.npy");

    LOG(info) << "X4PhysicsVector<double>::_hc_eVnm " << X4PhysicsVector<double>::_hc_eVnm() ; 

}

void X4PhysicsVectorTest::test_digest()
{
    std::string dig = X4PhysicsVector<float>::Digest(pof) ; 

    LOG(info) << " dig " << dig ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    X4PhysicsVectorTest pvt  ; 
    pvt.test_units();
    pvt.test_digest(); 
    pvt.test_convert(); 
    pvt.test_g4interpolate(); 

    return 0 ; 
}
