#include "OPTICKS_LOG.hh"
#include "OKConf.hh"
#include "SPath.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "NP.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    NP* a = NP::Make<double>(6); 
    double* a_v = a->values<double>(); 
    int g4version = OKConf::Geant4VersionInteger() ; 

    a_v[0] = h_Planck ; 
    a_v[1] = c_light ; 
    a_v[2] = h_Planck*c_light ; 
    a_v[3] = h_Planck*c_light/nm ; 
    a_v[4] = nm/(h_Planck*c_light) ; 
    a_v[5] = nm ; 

    std::stringstream ss ; 
    ss << "h_Planck" << std::endl ; 
    ss << "c_light" << std::endl ; 
    ss << "h_Planck*c_light" << std::endl ; 
    ss << "h_Planck*c_light/nm" << std::endl ; 
    ss << "nm/(h_Planck*c_light)" << std::endl ; 
    ss << "nm" << std::endl ; 
    a->meta = ss.str(); 

    std::stringstream nn ; 
    nn << g4version << ".npy" ; 
    std::string s = nn.str(); 
    const char* name = s.c_str(); 

    const char* fold = SPath::Resolve("$TMP/X4PhysicalConstantsTest", 2); // 2:dirpath 
    a->save( fold, name ); 

    return 0 ; 
}


