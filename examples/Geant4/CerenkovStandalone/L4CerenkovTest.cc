
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <csignal>

#include "NP.hh"

#include "Randomize.hh"
#include "G4MaterialPropertyVector.hh"
#include "G4Material.hh"
#include "G4PhysicsTable.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"




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

/**


::

    g4-cls G4MaterialPropertyVector
    g4-cls G4PhysicsOrderedFreeVector

**/

struct L4Cerenkov
{
    static NP* LoadArray(const char* kdpath);
    static G4MaterialPropertyVector* MakeProperty(const NP* a);
    static G4MaterialPropertyVector* MakeWaterRI(); 
    static G4Material* MakeMaterial(G4MaterialPropertyVector* rindex) ; 

    L4Cerenkov(G4MaterialPropertyVector* rindex); 

    const G4Material*          aMaterial ; 
    G4MaterialPropertiesTable* aMaterialPropertiesTable ;
    G4MaterialPropertyVector*  Rindex ; 
    G4PhysicsTable*            thePhysicsTable ; 


    void BuildThePhysicsTable(); 
    G4double GetAverageNumberOfPhotons(const G4double charge, const G4double beta) ; 
    void SampleWavelengths(G4double BetaInverse ) ;
    bool looping_condition(unsigned& count);
    void BetaInverseScan();


    int append_group ; 

    static constexpr unsigned itemsize0 = 16 ; 
    std::vector<std::string> names0 ; 
    std::vector<double>      dbg0 ; 

    static constexpr unsigned itemsize1 = 8 ; 
    std::vector<std::string> names1 ; 
    std::vector<double>      dbg1 ; 

    static constexpr unsigned itemsize2 = 8 ; 
    std::vector<std::string> names2 ; 
    std::vector<double>      dbg2 ; 



    std::vector<double>      wavelengths ; 


    unsigned branch ; 
    void dump(); 
    void dump(const std::vector<double>& dbg, const std::vector<std::string>& names, unsigned itemsize); 

    void WriteDbg(const char* dir, unsigned group);
    void Write(const char* dir);

    void append( double x, const char* name );
    void append( unsigned x, unsigned y, const char* name ); 
};


NP* L4Cerenkov::LoadArray(const char* kdpath)
{
    const char* keydir = getenv("OPTICKS_KEYDIR") ; 
    assert( keydir ); 
    std::stringstream ss ; 
    ss << keydir << "/" << kdpath ;  
    std::string s = ss.str(); 
    const char* path = s.c_str(); 
    std::cout << "L4Cerenkov::LoadArray " << path << std::endl ; 
    NP* a = NP::Load(path); 
    return a ; 
}

G4MaterialPropertyVector* L4Cerenkov::MakeProperty(const NP* a)
{
    unsigned nv = a->num_values() ; 
    std::cout << "a " << a->desc() << " num_values " << nv << std::endl ; 
    const double* vv = a->values<double>() ; 

    assert( nv %  2 == 0 ); 
    unsigned entries = nv/2 ; 
    std::vector<double> e(entries, 0.); 
    std::vector<double> v(entries, 0.); 

    for(unsigned i=0 ; i < entries ; i++)
    {
        e[i] = vv[2*i+0] ; 
        v[i] = vv[2*i+1] ; 
        std::cout 
            << " e[i]/eV " << std::fixed << std::setw(10) << std::setprecision(4) << e[i]/eV
            << " v[i] " << std::fixed << std::setw(10) << std::setprecision(4) << v[i] 
            << std::endl 
            ;     
    }
    G4MaterialPropertyVector* mpv = new G4MaterialPropertyVector(e.data(), v.data(), entries ); 
    return mpv ; 
}




G4MaterialPropertyVector* L4Cerenkov::MakeWaterRI()
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

G4Material* L4Cerenkov::MakeMaterial(G4MaterialPropertyVector* rindex)  // static
{
    // its Water, but that makes no difference for Cerenkov 
    // the only thing that matters us the rindex property
    G4double a, z, density;
    G4int nelements;
    G4Element* O = new G4Element("Oxygen"  , "O", z=8 , a=16.00*CLHEP::g/CLHEP::mole);
    G4Element* H = new G4Element("Hydrogen", "H", z=1 , a=1.01*CLHEP::g/CLHEP::mole);
    G4Material* mat = new G4Material("Water", density= 1.0*CLHEP::g/CLHEP::cm3, nelements=2);
    mat->AddElement(H, 2);
    mat->AddElement(O, 1);

    rindex->SetSpline(false);
    //rindex->SetSpline(true);

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    mpt->AddProperty("RINDEX", rindex );   
    mat->SetMaterialPropertiesTable(mpt) ;
    return mat ; 
}


L4Cerenkov::L4Cerenkov(G4MaterialPropertyVector* rindex)
    :
    aMaterial(MakeMaterial(rindex)),
    aMaterialPropertiesTable(aMaterial ? aMaterial->GetMaterialPropertiesTable() : nullptr),
    Rindex(aMaterialPropertiesTable ? aMaterialPropertiesTable->GetProperty("RINDEX") : nullptr),
    thePhysicsTable(nullptr),
    append_group(-1)
{
    assert(aMaterialPropertiesTable) ; 
    assert(Rindex) ;
    assert(Rindex == rindex ) ;
    //G4cout << *Rindex << G4endl ; 
    BuildThePhysicsTable(); 
}


void L4Cerenkov::append( double x, const char* name )
{
    if( append_group == 0 )
    {
        dbg0.push_back(x);  
        if( names0.size() <= itemsize0 ) names0.push_back(name); 
    }
    else if( append_group == 1 )
    {
        dbg1.push_back(x);  
        if( names1.size() <= itemsize1 ) names1.push_back(name); 
    }
    else if( append_group == 2 )
    {
        dbg2.push_back(x);  
        if( names2.size() <= itemsize2 ) names2.push_back(name); 
    }
} 
void L4Cerenkov::append( unsigned x, unsigned y, const char* name )
{
    assert( sizeof(unsigned)*2 == sizeof(double) ); 
    DUU duu ; 
    duu.uu.x = x ; 
    duu.uu.y = y ;            // union trickery to place two unsigned into the slot of a double  
    append(duu.d, name); 
}

void L4Cerenkov::BuildThePhysicsTable()
{
	if (thePhysicsTable) return;

	const G4MaterialTable* theMaterialTable= G4Material::GetMaterialTable(); 
	G4int numOfMaterials = G4Material::GetNumberOfMaterials();
    assert( numOfMaterials == 1 ); 

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
L4Cerenkov::GetAverageNumberOfPhotons(const G4double charge, const G4double beta)
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

    // g4-cls G4PhysicsOrderedFreeVector
    // these are the values from the first and last bin, NOT the miniumum and maximum value across the range 
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



void L4Cerenkov::dump()
{
    switch(append_group)
    {
       case 0: dump( dbg0, names0, itemsize0 ) ; break ; 
       case 1: dump( dbg1, names1, itemsize1 ) ; break ; 
       case 2: dump( dbg2, names2, itemsize2 ) ; break ; 
    }
}

void L4Cerenkov::dump(const std::vector<double>& dbg, const std::vector<std::string>& names, unsigned itemsize)
{
    unsigned n = dbg.size() ; 
    for( unsigned i=0 ; i < n ; i++)
    {
        const std::string& name = names[i % itemsize] ; 
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

bool L4Cerenkov::looping_condition(unsigned& count)
{
    count += 1 ; 
    return true ; 
}


void L4Cerenkov::SampleWavelengths(G4double BetaInverse ) 
{
    G4double beta   = 1./BetaInverse ; 

    G4double Pmin = Rindex->GetMinLowEdgeEnergy();
    G4double Pmax = Rindex->GetMaxLowEdgeEnergy();
    G4double dp = Pmax - Pmin;

    G4double nMax0 = Rindex->GetMaxValue();
    G4double nMax = 0. ; 
    // extract from "jcv G4Cerenkov_modified" to get the real maximum rindex over the domain 
    for (size_t i = 0; i < Rindex->GetVectorLength(); ++i) if ((*Rindex)[i]>nMax) nMax = (*Rindex)[i];

    std::cout 
        << "[ L4Cerenkov::SampleWavelengths" 
        << " BetaInverse " << std::fixed << std::setw(10) << std::setprecision(4) << BetaInverse 
        << " nMax0 "       << std::fixed << std::setw(10) << std::setprecision(4) << nMax0
        << " nMax "        << std::fixed << std::setw(10) << std::setprecision(4) << nMax
        << std::endl 
        ; 


    G4double maxCos = BetaInverse / nMax;
    G4double maxSin2 = (1.0 - maxCos) * (1.0 + maxCos);

    G4double charge = 1. ; 
    G4double MeanNumberOfPhotons = GetAverageNumberOfPhotons(charge,beta );

    wavelengths.resize(1000000, 0.f); 

    append_group = 2 ; 

    append( BetaInverse, "BetaInverse" ); 
    append( beta       , "beta" ); 
    append( Pmin       , "Pmin" ); 
    append( Pmax       , "Pmax" ); 

    append( nMax       , "nMax" ); 
    append( maxCos     , "maxCos" ); 
    append( maxSin2    , "maxSin2" ); 
    append( MeanNumberOfPhotons    , "MeanNumberOfPhotons" ); 

    append_group = 1 ; 

    for(unsigned i=0 ; i < wavelengths.size() ; i++)
    {
        // Determine photon energy

        G4double rand;
        G4double sampledEnergy, sampledRI;
        G4double cosTheta, sin2Theta;

        // sample an energy

        unsigned head_count = 0 ; 
        unsigned tail_count = 0 ; 
        unsigned continue_count = 0 ; 
        unsigned condition_count = 0 ; 

        do {

            head_count += 1 ; 
            rand = G4UniformRand();
            sampledEnergy = Pmin + rand * dp;
            sampledRI = Rindex->Value(sampledEnergy);

            // G4Cerenkov_modified attempts to continue 
            // to the next turn of the  while loop 
            //  
            // That looks like a bug as as could
            // in prinipal result in cosTheta sin2Theta  
            // being used uninitialized. 
            //
            //
            // check if n(E) > 1/beta
            if (sampledRI < BetaInverse) {
               continue_count += 1 ; 
               continue;
            }   

            tail_count += 1 ; 
            cosTheta = BetaInverse / sampledRI;

            sin2Theta = (1.0 - cosTheta)*(1.0 + cosTheta);
            rand = G4UniformRand();

            // Loop checking, 07-Aug-2015, Vladimir Ivanchenko
        } while ( looping_condition(condition_count) && rand*maxSin2 > sin2Theta  );

        G4double sampledWavelength_nm = h_Planck*c_light/sampledEnergy/nm ; 

        append( sampledEnergy ,           "sampledEnergy" ); 
        append( sampledWavelength_nm ,    "sampledWavelength" ); 
        append( sampledRI ,               "sampledRI" ); 
        append( cosTheta ,                "cosTheta" ); 

        append( sin2Theta ,               "sin2Theta" ); 
        append( head_count ,     tail_count,       "head_tail" ); 
        append( continue_count , condition_count,  "continue_condition" ); 
        append( BetaInverse , "BetaInverse" ); 

        wavelengths[i] = sampledWavelength_nm ; 
    }

    append_group = -1 ; 

    std::cout 
        << "] L4Cerenkov::SampleWavelengths BetaInverse " 
        << std::fixed << std::setw(10) << std::setprecision(4) << BetaInverse 
        << std::endl 
        ; 
}


void L4Cerenkov::BetaInverseScan()
{
    append_group = 0 ; 

	G4double nMin = 1000. ; 
    for (size_t i = 0; i < Rindex->GetVectorLength(); ++i) if ((*Rindex)[i]<nMin) nMin = (*Rindex)[i];

	G4double nMax = 0. ;
    for (size_t i = 0; i < Rindex->GetVectorLength(); ++i) if ((*Rindex)[i]>nMax) nMax = (*Rindex)[i];

    std::cout 
        << "[ BetaInverseScan " 
        << " nMin " << nMin 
        << " nMax " << nMax 
        << std::endl 
        ; 

    const G4double charge = 1.0 ; 
    G4double BetaInverse ; 

    unsigned count = 0 ; 

    for( BetaInverse=1.0 ; BetaInverse < nMax+0.1 ; BetaInverse += 0.001 )
    {
        G4double beta   = 1./BetaInverse ; 
        G4double gamma = 1./sqrt(1.-(beta*beta)) ; 

        G4double AverageNumberOfPhotons = GetAverageNumberOfPhotons(charge, beta ); 
        append(gamma, "gamma" );  
        append(0., "Padding0"); 

        if(count % 100 == 0 ) std::cout 
            << " count " << std::setw(4) << count   
            << " gamma " << std::setw(15) << std::fixed << std::setprecision(10) << gamma 
            << " BetaInverse " << std::setw(15) << std::fixed << std::setprecision(10) << BetaInverse
            << " beta " << std::setw(15) << std::fixed << std::setprecision(10) << beta
            << " l4c.branch " << std::setw(5) << branch 
            << " AverageNumberOfPhotons " << std::setw(15) << std::fixed << std::setprecision(10) << AverageNumberOfPhotons 
            << std::endl 
            ;

        count += 1 ; 
    }
    std::cout << "] BetaInverseScan dbg0.size " << dbg0.size() << " dbg0.size % 16 : " << dbg0.size() % 16 << std::endl ;
    append_group = -1 ; 
}


void L4Cerenkov::WriteDbg(const char* dir, unsigned group)
{

    unsigned itemsize = 0 ; 
    const std::vector<double>* dbg = nullptr ;     
    const std::vector<std::string>* names = nullptr ;     

    if( group == 0 )
    {
        itemsize = itemsize0 ; 
        dbg = &dbg0 ; 
        names = &names0 ;
 
    }
    else if( group == 1 )
    {
        itemsize = itemsize1 ; 
        dbg = &dbg1 ; 
        names = &names1 ;
    }
    else if( group == 2 )
    {
        itemsize = itemsize2 ; 
        dbg = &dbg2 ; 
        names = &names2 ;
    }
  

    bool expected_size = dbg->size() % itemsize == 0  ; 
    unsigned ni = dbg->size() / itemsize  ; 

    if(!expected_size)
    {
       std::cout 
           << " UNEXPECTED SIZE "
           << " dbg->size " << dbg->size()
           << " itemsize " << itemsize 
           << " ni " << ni
           << std::endl 
           ;
    }
    assert( expected_size ); 

    if( group == 0 )
    {
        NP::Write(     dir, "BetaInverseScan.npy", dbg->data(), ni, 4, 4 ); 
        NP::WriteNames(dir, "BetaInverseScan.txt", *names, itemsize ); 
    }
    else if( group == 1 )
    {
        NP::Write(     dir, "SampleWavelengths.npy", dbg->data(), ni, 2, 4 ); 
        NP::WriteNames(dir, "SampleWavelengths.txt", *names, itemsize ); 
    }
    else if( group == 2 )
    {
        NP::Write(     dir, "Params.npy", dbg->data(), itemsize ); 
        NP::WriteNames(dir, "Params.txt", *names, itemsize ); 
    }
}


void L4Cerenkov::Write(const char* dir  )
{
    WriteDbg(dir, 0); 
    WriteDbg(dir, 1); 
    WriteDbg(dir, 2); 

    NP::Write(     dir, "SampleWavelengths_.npy", wavelengths.data(), wavelengths.size() ); 
}


int main(int argc, char** argv)
{
    double BetaInverse = argc > 1 ? std::stod(argv[1]) : 1.02 ;  

    const char* path = "GScintillatorLib/LS_ori/RINDEX.npy" ; 
    std::cout 
         << " path " << path 
         << " BetaInverse " << std::setw(10) << std::fixed << std::setprecision(4) << BetaInverse 
         << std::endl 
         ;


    const char* FOLD = "/tmp/L4CerenkovTest" ; 

    G4MaterialPropertyVector* rindex = nullptr ; 
    if( strcmp(path, "Water") == 0 )
    {
        rindex = L4Cerenkov::MakeWaterRI(); 
    }
    else
    {
        std::cout << "load from " << path << std::endl ;  
        NP* a = L4Cerenkov::LoadArray(path) ; 
        a->save(FOLD, "RINDEX.npy");  
        rindex = L4Cerenkov::MakeProperty(a); 
    }
    assert( rindex );  

    L4Cerenkov l4c(rindex) ; 

    l4c.BetaInverseScan(); 

    l4c.SampleWavelengths(BetaInverse); 

    l4c.Write(FOLD); 

    return 0 ; 
}
