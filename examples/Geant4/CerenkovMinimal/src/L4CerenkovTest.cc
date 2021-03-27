
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "NP.hh"

#include "G4MaterialPropertyVector.hh"
#include "G4Material.hh"
#include "G4PhysicsTable.hh"
#include "G4SystemOfUnits.hh"

struct DetectorConstruction
{
    static G4MaterialPropertyVector* MakeWaterRI() ; 
    static G4Material* MakeWater();
    static void AddProperty( G4Material* mat , const char* name, G4MaterialPropertyVector* mpv ) ;
};

G4MaterialPropertyVector* DetectorConstruction::MakeWaterRI()
{
    using CLHEP::eV ; 
    G4double photonEnergy[] =
                { 2.034*eV, 2.068*eV, 2.103*eV, 2.139*eV,
                  2.177*eV, 2.216*eV, 2.256*eV, 2.298*eV,
                  2.341*eV, 2.386*eV, 2.433*eV, 2.481*eV,
                  2.532*eV, 2.585*eV, 2.640*eV, 2.697*eV,
                  2.757*eV, 2.820*eV, 2.885*eV, 2.954*eV,
                  3.026*eV, 3.102*eV, 3.181*eV, 3.265*eV,
                  3.353*eV, 3.446*eV, 3.545*eV, 3.649*eV,
                  3.760*eV, 3.877*eV, 4.002*eV, 4.136*eV };

    const G4int nEntries = sizeof(photonEnergy)/sizeof(G4double);

    G4double refractiveIndex[] =
            { 1.3435, 1.344,  1.3445, 1.345,  1.3455,
              1.346,  1.3465, 1.347,  1.3475, 1.348,
              1.3485, 1.3492, 1.35,   1.3505, 1.351,
              1.3518, 1.3522, 1.3530, 1.3535, 1.354,
              1.3545, 1.355,  1.3555, 1.356,  1.3568,
              1.3572, 1.358,  1.3585, 1.359,  1.3595,
              1.36,   1.3608};

    assert(sizeof(refractiveIndex) == sizeof(photonEnergy));
    return new G4MaterialPropertyVector(photonEnergy, refractiveIndex,nEntries ); 
}


G4Material* DetectorConstruction::MakeWater()
{
    G4double a, z, density;
    G4int nelements;
    G4Element* O = new G4Element("Oxygen"  , "O", z=8 , a=16.00*CLHEP::g/CLHEP::mole);
    G4Element* H = new G4Element("Hydrogen", "H", z=1 , a=1.01*CLHEP::g/CLHEP::mole);
    G4Material* mat = new G4Material("Water", density= 1.0*CLHEP::g/CLHEP::cm3, nelements=2);
    mat->AddElement(H, 2);
    mat->AddElement(O, 1);

    G4MaterialPropertyVector* ri = MakeWaterRI() ; 
    ri->SetSpline(false);
    //ri->SetSpline(true);

    AddProperty( mat, "RINDEX" , ri );  
    return mat ; 
}


void DetectorConstruction::AddProperty( G4Material* mat , const char* name, G4MaterialPropertyVector* mpv )
{
    G4MaterialPropertiesTable* mpt = mat->GetMaterialPropertiesTable(); 
    if( mpt == NULL ) mpt = new G4MaterialPropertiesTable();
    mpt->AddProperty(name, mpv );   
    mat->SetMaterialPropertiesTable(mpt) ;
}  



struct UU
{
   unsigned x ; 
   unsigned y ; 
};

union DUU
{
   double d ; 
   UU     uu ; 
};


struct L4Cerenkov
{
    void BuildThePhysicsTable(); 

    G4double GetAverageNumberOfPhotons(const G4double charge,
                                  const G4double beta, 
                      const G4Material* aMaterial,
                      G4MaterialPropertyVector* Rindex) ; 

    G4PhysicsTable* thePhysicsTable = nullptr ; 

    std::vector<std::string> names ; 
    std::vector<double> dbg ; 
    unsigned branch ; 

    void Dump(); 
    void Write(const char* dir, const char* npy, const char* txt );

    void append( double x, const char* name );
    void append( unsigned x, unsigned y, const char* name ); 

};


void L4Cerenkov::append( double x, const char* name )
{
    dbg.push_back(x); 
    names.push_back(name); 
} 
void L4Cerenkov::append( unsigned x, unsigned y, const char* name )
{
    assert( sizeof(unsigned)*2 == sizeof(double) ); 
    DUU duu ; 
    duu.uu.x = x ; 
    duu.uu.y = y ;            // union trickery to place two unsigned into the slot of a double  
    dbg.push_back(duu.d);  
    names.push_back(name); 
}







void L4Cerenkov::BuildThePhysicsTable()
{
	if (thePhysicsTable) return;

	const G4MaterialTable* theMaterialTable=
	 		       G4Material::GetMaterialTable();
	G4int numOfMaterials = G4Material::GetNumberOfMaterials();

	// create new physics table
	
	thePhysicsTable = new G4PhysicsTable(numOfMaterials);

	// loop for materials

	for (G4int i=0 ; i < numOfMaterials; i++)
	{
	        G4PhysicsOrderedFreeVector* aPhysicsOrderedFreeVector = 0;

		// Retrieve vector of refraction indices for the material
		// from the material's optical properties table 

		G4Material* aMaterial = (*theMaterialTable)[i];

		G4MaterialPropertiesTable* aMaterialPropertiesTable =
				aMaterial->GetMaterialPropertiesTable();

		if (aMaterialPropertiesTable) {

		   aPhysicsOrderedFreeVector = new G4PhysicsOrderedFreeVector();
		   G4MaterialPropertyVector* theRefractionIndexVector = 
		    	   aMaterialPropertiesTable->GetProperty("RINDEX");

		   if (theRefractionIndexVector) {
		
		      // Retrieve the first refraction index in vector
		      // of (photon energy, refraction index) pairs 

                      G4double currentRI = (*theRefractionIndexVector)[0];

		      if (currentRI > 1.0) {

			 // Create first (photon energy, Cerenkov Integral)
			 // pair  

                         G4double currentPM = theRefractionIndexVector->
                                                 Energy(0);
			 G4double currentCAI = 0.0;

			 aPhysicsOrderedFreeVector->
			 	 InsertValues(currentPM , currentCAI);

			 // Set previous values to current ones prior to loop

			 G4double prevPM  = currentPM;
			 G4double prevCAI = currentCAI;
                	 G4double prevRI  = currentRI;

			 // loop over all (photon energy, refraction index)
			 // pairs stored for this material  

                         for (size_t ii = 1;
                              ii < theRefractionIndexVector->GetVectorLength();
                              ++ii)
			 {
                                currentRI = (*theRefractionIndexVector)[ii];
                                currentPM = theRefractionIndexVector->Energy(ii);

				currentCAI = 0.5*(1.0/(prevRI*prevRI) +
					          1.0/(currentRI*currentRI));

				currentCAI = prevCAI + 
					     (currentPM - prevPM) * currentCAI;

				aPhysicsOrderedFreeVector->
				    InsertValues(currentPM, currentCAI);

				prevPM  = currentPM;
				prevCAI = currentCAI;
				prevRI  = currentRI;
			 }

		      }
		   }
		}

	// The Cerenkov integral for a given material
	// will be inserted in thePhysicsTable
	// according to the position of the material in
	// the material table. 

	thePhysicsTable->insertAt(i,aPhysicsOrderedFreeVector); 

	}
}



// GetAverageNumberOfPhotons
// -------------------------
// This routine computes the number of Cerenkov photons produced per
// GEANT-unit (millimeter) in the current medium. 
//             ^^^^^^^^^^

G4double 
L4Cerenkov::GetAverageNumberOfPhotons(const G4double charge,
                              const G4double beta, 
			      const G4Material* aMaterial,
			      G4MaterialPropertyVector* Rindex) 
{
    append(charge, "charge"); 
    append(beta,   "beta"); 

	const G4double Rfact = 369.81/(eV * cm);
    append(Rfact, "Rfact"); 

    //    if(beta <= 0.0)return 0.0;
    assert( beta >= 0.0 ); 

        G4double BetaInverse = 1./beta;
    append(BetaInverse, "BetaInverse"); 

	// Vectors used in computation of Cerenkov Angle Integral:
	// 	- Refraction Indices for the current material
	//	- new G4PhysicsOrderedFreeVector allocated to hold CAI's
 
	G4int materialIndex = aMaterial->GetIndex();

	// Retrieve the Cerenkov Angle Integrals for this material  

	G4PhysicsOrderedFreeVector* CerenkovAngleIntegrals =
	(G4PhysicsOrderedFreeVector*)((*thePhysicsTable)(materialIndex));

    //    if(!(CerenkovAngleIntegrals->IsFilledVectorExist()))return 0.0;
    assert( CerenkovAngleIntegrals->IsFilledVectorExist() );  


	// Min and Max photon energies 
	G4double Pmin = Rindex->GetMinLowEdgeEnergy();
	G4double Pmax = Rindex->GetMaxLowEdgeEnergy();

    append(Pmin, "Pmin"); 
    append(Pmax, "Pmax"); 

	// Min and Max Refraction Indices 
	G4double nMin = Rindex->GetMinValue();	
	G4double nMax = Rindex->GetMaxValue();

    append(nMin, "nMin"); 
    append(nMax, "nMax"); 


	// Max Cerenkov Angle Integral 
	G4double CAImax = CerenkovAngleIntegrals->GetMaxValue();
    append(CAImax, "CAImax"); 

	G4double dp, ge;

	// If n(Pmax) < 1/Beta -- no photons generated 


    G4double CAImin = 0. ; 

	if (nMax < BetaInverse) {
		dp = 0;
		ge = 0;
        branch = 0 ; 
	} 

	// otherwise if n(Pmin) >= 1/Beta -- photons generated  

	else if (nMin > BetaInverse) {
		dp = Pmax - Pmin;	
		ge = CAImax; 
        branch = 1 ; 
	} 

	// If n(Pmin) < 1/Beta, and n(Pmax) >= 1/Beta, then
	// we need to find a P such that the value of n(P) == 1/Beta.
	// Interpolation is performed by the GetEnergy() and
	// Value() methods of the G4MaterialPropertiesTable and
	// the GetValue() method of G4PhysicsVector.  

	else {
        branch = 2 ; 
		Pmin = Rindex->GetEnergy(BetaInverse);
		dp = Pmax - Pmin;

		// need boolean for current implementation of G4PhysicsVector
		// ==> being phased out
		G4bool isOutRange;
		CAImin = CerenkovAngleIntegrals->
                                  GetValue(Pmin, isOutRange);
		ge = CAImax - CAImin;

#ifdef WITH_DUMP
		if (verboseLevel>0) {
			G4cout << "CAImin = " << CAImin << G4endl;
			G4cout << "ge = " << ge << G4endl;
		}
#endif
	}

    append(materialIndex, branch, "materialIndex_branch"); 
    append(dp, "dp"); 
    append(ge, "ge"); 
    append(CAImin, "CAImin"); 
	
	// Calculate number of photons 
	G4double NumPhotons = Rfact * charge/eplus * charge/eplus *
                                 (dp - ge * BetaInverse*BetaInverse);


    append(NumPhotons, "NumPhotons"); 

	return NumPhotons;		
}

void L4Cerenkov::Dump()
{
    assert( dbg.size() == names.size() ); 
    unsigned n = dbg.size() ; 
    for( unsigned i=0 ; i < n ; i++)
    {
        const std::string& name = names[i] ; 
        double value = dbg[i] ; 
        int w = 25 ; 
        std::cout 
            << std::setw(3) << i
            << " : " 
            << std::setw(w) << name 
            << " : "
            << std::fixed << std::setprecision(5) << value 
            << std::endl 
            ;
 
    }
}

void L4Cerenkov::Write(const char* dir, const char* npy, const char* txt )
{
    unsigned itemsize = 16 ; 
    assert( dbg.size() % itemsize == 0 ); 

    unsigned ni = dbg.size() / itemsize  ; 

    NP::Write(dir, npy, dbg.data(), ni, 4, 4 ); 

    std::stringstream ss ; 
    ss << dir << "/" << txt ; 
    std::string s = ss.str() ; 

    std::ofstream stream(s.c_str(), std::ios::out|std::ios::binary);
    for( unsigned i=0 ; i < itemsize ; i++) 
    {
        std::cout << " write " << std::setw(3) << i << " : " << names[i] << std::endl ; 
        stream << names[i] << std::endl ; 
    }
    stream.close(); 
}



int main(int argc, char** argv)
{
    G4Material* water = DetectorConstruction::MakeWater(); 
    G4cout << *water << G4endl ; 

    L4Cerenkov l4c ; 
    l4c.BuildThePhysicsTable(); 

    const G4Material*  aMaterial = water ; 

    G4MaterialPropertiesTable* aMaterialPropertiesTable = aMaterial->GetMaterialPropertiesTable();
    assert(aMaterialPropertiesTable) ; 
    G4MaterialPropertyVector* Rindex = aMaterialPropertiesTable->GetProperty("RINDEX"); 
    G4cout << *Rindex << G4endl ; 

   	// Min and Max Refraction Indices 
	G4double nMin = Rindex->GetMinValue();	
	G4double nMax = Rindex->GetMaxValue();

    std::cout << " nMin " << nMin << std::endl ; 
    std::cout << " nMax " << nMax << std::endl ; 

    const G4double charge = 1.0 ; 
    G4double BetaInverse ; 
    for( BetaInverse=1.0 ; BetaInverse < nMax+0.1 ; BetaInverse += 0.001 )
    {
        G4double beta   = 1./BetaInverse ; 
        G4double gamma = 1./sqrt(1.-(beta*beta)) ; 

        G4double AverageNumberOfPhotons = l4c.GetAverageNumberOfPhotons(charge, beta, aMaterial, Rindex); 
        l4c.append(gamma, "gamma" );  
        l4c.append(0., "Padding0"); 
        std::cout 
            << " gamma " << std::setw(15) << std::fixed << std::setprecision(10) << gamma 
            << " BetaInverse " << std::setw(15) << std::fixed << std::setprecision(10) << BetaInverse
            << " beta " << std::setw(15) << std::fixed << std::setprecision(10) << beta
            << " l4c.branch " << std::setw(5) << l4c.branch 
            << " AverageNumberOfPhotons " << std::setw(15) << std::fixed << std::setprecision(10) << AverageNumberOfPhotons 
            << std::endl 
            ;
    }

    l4c.Write("/tmp", "L4CerenkovTest.npy", "L4CerenkovTest.txt"); 

    return 0 ; 
}
