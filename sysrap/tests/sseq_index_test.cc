// ~/opticks/sysrap/tests/sseq_index_test.sh 

//#include "spath.h"
#include "NP.hh"
#include "sseq_index.h"

int main()
{
    //const char* path = spath::Resolve("$TMP/GEOM/$GEOM/$EXECUTABLE/ALL$VERSION/$EVT/seq.npy") ; 
    const char* path = "$TMP/GEOM/$GEOM/$EXECUTABLE/ALL$VERSION/$EVT/seq.npy" ; 
    NP* seq = NP::LoadIfExists(path); 
    std::cout << " path " << path << " seq " << ( seq ? seq->sstr() : "-" ) << std::endl ; 
    if(!seq) return 0 ;  

    sseq_index qx(seq); 
    std::cout << qx.desc(100) << std::endl ; 

    return 0 ; 
}
// ~/opticks/sysrap/tests/sseq_index_test.sh 



