#include <iostream>
#include <iomanip>
#include <vector>

#include "G4SystemOfUnits.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4Types.hh"
#include "Randomize.hh"

struct ScintTest
{
    G4MaterialPropertiesTable* LSMPT ; 

    ScintTest();
    void init(); 

    G4MaterialPropertyVector* getRatiosProperty(const G4String& aParticleName) const ;
    void          getSplits(std::vector<int>& splits, G4int NumTracks, const G4String& aParticleName ) const ;
    std::string desc( const std::vector<int>& splits, G4int NumTracks,  const G4String& aParticleName ) const ;

};

ScintTest::ScintTest()
    :
    LSMPT(new G4MaterialPropertiesTable)
{
    init(); 
}


void ScintTest::init()
{
   // jcv LSExpDetectorConstructionMaterial OpticalProperty DsG4Scintillation 

  double OpticalYieldRatio[2] = {1.0 , 0 };
  double OpticalTimeConstant[2] = {1.50*ns ,1.50*ns};
  double GammaYieldRatio[3] = {0.799, 0.171, 0.03};
  double GammaTimeConstant[3] = { 4.93*ns , 20.6*ns, 190*ns};
  double AlphaYieldRatio[3] = {0.65, 0.2275, 0.1225};
  double AlphaTimeConstant[3] = { 4.93*ns ,35.0*ns, 220*ns};
  double NeutronYieldRatio[3] = { 0.65, 0.231, 0.119};
  double NeutronTimeConstant[3] = { 4.93*ns, 34*ns ,220*ns};

    LSMPT->AddProperty("OpticalCONSTANT",OpticalTimeConstant,OpticalYieldRatio,2);
    LSMPT->AddProperty("GammaCONSTANT", GammaTimeConstant , GammaYieldRatio, 3);
    LSMPT->AddProperty("AlphaCONSTANT", AlphaTimeConstant , AlphaYieldRatio, 3);
    LSMPT->AddProperty("NeutronCONSTANT", NeutronTimeConstant , NeutronYieldRatio, 3);

}


G4MaterialPropertyVector* ScintTest::getRatiosProperty(const G4String& aParticleName) const 
{
   G4MaterialPropertiesTable* aMaterialPropertiesTable = LSMPT  ; 
   G4MaterialPropertyVector* Ratio_timeconstant = 0 ;
    if (aParticleName == "opticalphoton") {
      Ratio_timeconstant = aMaterialPropertiesTable->GetProperty("OpticalCONSTANT");
    }
    else if(aParticleName == "gamma" || aParticleName == "e+" || aParticleName == "e-") {
      Ratio_timeconstant = aMaterialPropertiesTable->GetProperty("GammaCONSTANT");
    }
    else if(aParticleName == "alpha") {
      Ratio_timeconstant = aMaterialPropertiesTable->GetProperty("AlphaCONSTANT");
    }
    else {
      Ratio_timeconstant = aMaterialPropertiesTable->GetProperty("NeutronCONSTANT");
    }

    return Ratio_timeconstant ;  
}



void ScintTest::getSplits(std::vector<int>& splits, G4int NumTracks, const G4String& aParticleName ) const 
{
    G4MaterialPropertyVector* Ratio_timeconstant = getRatiosProperty(aParticleName );    
    size_t nscnt = Ratio_timeconstant->GetVectorLength();

    splits.resize(nscnt); 
    std::vector<G4int>& m_Num = splits ; 

    m_Num.clear();
    for(G4int i = 0 ; i < NumTracks ; i++){
       G4double p = G4UniformRand();
       G4double p_count = 0;
       for(G4int j = 0 ; j < nscnt; j++)
       {   
            p_count += (*Ratio_timeconstant)[j] ;
            if( p < p_count ){
               m_Num[j]++ ;
               break;
            }   
        }   
     }   
}



std::string ScintTest::desc( const std::vector<int>& splits, G4int NumTracks,  const G4String& aParticleName ) const
{
    G4MaterialPropertyVector* Ratio_timeconstant = getRatiosProperty(aParticleName );    
    size_t nscnt = Ratio_timeconstant->GetVectorLength();

    std::stringstream ss ; 
    ss 
        << aParticleName
        << " NumTracks " << NumTracks 
        << " nscnt " << nscnt 
        << std::endl 
        ; 

    const std::vector<G4int>& m_Num = splits ; 

    G4int tot = 0 ; 
    G4double sum_rat = 0 ; 
    for(G4int j = 0 ; j < nscnt; j++)
    {
       G4double rat = (*Ratio_timeconstant)[j] ; 


       G4double ScintillationTime = Ratio_timeconstant->Energy(j);

       ss
           << " j " << std::setw(3) << j 
           << " Num_j " << std::setw(5) << m_Num[j]
           << " rat " << std::setw(10) << std::fixed << std::setprecision(3) << rat
           << " ScintillationTime " << std::setw(10) << std::fixed << std::setprecision(3) << ScintillationTime/ns << " ns " 
           << std::endl 
           ; 
        tot += m_Num[j];  
        sum_rat += rat ; 
    }
    ss 
        << " TOTAL_Num  " << std::setw(5) << tot 
        << " sum_rat " << std::setw(10) << std::fixed << std::setprecision(3) << sum_rat
        << std::endl 
        ; 
 
    return ss.str(); 
}


int main(int argc, char** argv)
{
    ScintTest st ; 
    std::vector<int> splits ; 


    std::vector<std::string> _aParticleName = {"opticalphoton", "gamma", "alpha", "neutron" } ; 
    std::vector<G4int> _NumTracks = { 1, 100, 1000, 10000 } ; 

    for(unsigned p=0 ; p < _aParticleName.size() ; p++)
    {
        const std::string& aParticleName = _aParticleName[p] ;  
   
        std::cout << aParticleName << std::endl << std::endl ; 
 
        for(unsigned n=0 ; n < _NumTracks.size() ; n++)
        {   
            int NumTracks = _NumTracks[n] ; 
            st.getSplits(splits, NumTracks, aParticleName); 
            std::cout << st.desc(splits, NumTracks, aParticleName) << std::endl ;  
        }

    }

    return 0 ; 
}
