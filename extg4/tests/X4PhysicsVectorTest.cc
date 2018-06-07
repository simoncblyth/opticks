
#include "G4PhysicsOrderedFreeVector.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include "GProperty.hh"

#include "X4PhysicsVector.hh"
#include "OPTICKS_LOG.hh"


struct X4PhysicsVectorTest 
{
    size_t    veclen ; 
    G4double* energies ; 
    G4double* values   ;

    X4PhysicsVectorTest();

    void visibleSpectrum();
    void opNovice();

    G4PhysicsOrderedFreeVector* make_pof()
    {
        return new G4PhysicsOrderedFreeVector(energies, values, veclen) ; 
    } 
};




X4PhysicsVectorTest::X4PhysicsVectorTest()
    :
    veclen(0),
    energies(NULL),
    values(NULL)
{
    //visibleSpectrum();
    opNovice();
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


void test_units()
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


int main(int argc, char** argv)
{
    OPTICKS_LOG_COLOR__(argc, argv);
    
    //test_units();

    X4PhysicsVectorTest pvt  ; 

    G4PhysicsOrderedFreeVector* pof = pvt.make_pof(); 

    GProperty<float>* prop = X4PhysicsVector<float>::Convert(pof) ; 

    prop->SummaryV("prop", 50);

    prop->save("/tmp/prop.npy");

    return 0 ; 
}
