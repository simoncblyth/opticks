#include "G4Log.hh"
#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "SPath.hh"
#include "NP.hh"

const char* FOLD = SPath::Resolve("$TMP/U4LogTest", DIRPATH); 

int main(int argc, char** argv)
{
    unsigned ni = SSys::getenvunsigned("U4LogTest_ni", 1001) ; 
    unsigned nj = 5 ; 

    NP* a = NP::Make<double>(ni, nj ); 
    double* aa = a->values<double>();  

    for(unsigned i=0 ; i < ni ; i++)
    {
        double d =  double(i)/double(ni-1) ; 
        float  f = float(d) ; 

        double d0 = -1.*std::log( d ); 
        float  f0 = -1.f*std::log( f ); 

        double d4 = -1.*G4Log( d ) ; 
        float  f4 = -1.f*G4Logf( f ) ; 
        
        aa[nj*i+0] = d ; 
        aa[nj*i+1] = d0 ; 
        aa[nj*i+2] = f0 ; 
        aa[nj*i+3] = d4 ;
        aa[nj*i+4] = f4 ; 
    }

    a->save(FOLD, "a.npy") ; 
    return 0 ;  
}
