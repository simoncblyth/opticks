#include <cassert>
#include <vector>
#include <iomanip>
#include "CTrackInfo.hh"
#include "OPTICKS_LOG.hh"

void test_CTrackInfo()
{
     std::vector<std::string> checks = { " C0", " S0", "RS0", " C10", " S10", "RS10", " S100000", " C1000000" } ; 

     for(unsigned i=0 ; i < checks.size() ; i++)
     {   
         const char* chk = checks[i].c_str(); 
         char re = chk[0] ;
         char gentype = chk[1] ;
         unsigned photon_id = std::atoi(chk+2) ; 
         bool reemission = re == 'R' ;  

         CTrackInfo tkui(photon_id, gentype, reemission) ;
         const CTrk& trk = tkui.trk ; 

         LOG(info) 
             << " chk " << std::setw(20) << chk 
             << " re " << std::setw(2) << re 
             << " gentype " << std::setw(2) << gentype 
             << " photon_id " << std::setw(8) << photon_id 
             << " reemission  " << reemission 
             << " packed " << std::hex << std::setw(16) << trk.packed() << std::dec  
             << " tkui.GetType " << tkui.GetType()
             ;

         assert( trk.photon_id() == photon_id );  
         assert( trk.gentype() == gentype);  
         assert( trk.reemission() == reemission );  
     }   
}

int main(int argc, char** argv)
{  
    OPTICKS_LOG(argc, argv); 
    test_CTrackInfo();     
    return 0 ; 
}
