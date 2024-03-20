/**
sysrap/tests/sseq_index_test.sh 
=================================


::

    ~/opticks/sysrap/tests/sseq_index_test.sh 




**/


#include "NP.hh"
#include "sseq_index.h"

int main()
{
    const char* a_path = "$AFOLD/seq.npy" ; 
    const char* b_path = "$BFOLD/seq.npy" ; 
    NP* a_seq = NP::LoadIfExists(a_path); 
    NP* b_seq = NP::LoadIfExists(b_path); 
    std::cout << "a_path " << a_path << " " << ( a_seq ? a_seq->get_lpath() : "-" ) << " a_seq " << ( a_seq ? a_seq->sstr() : "-" ) << std::endl ; 
    std::cout << "b_path " << b_path << " " << ( b_seq ? b_seq->get_lpath() : "-" ) << " b_seq " << ( b_seq ? b_seq->sstr() : "-" ) << std::endl ; 
    if(!(a_seq && b_seq)) return 0 ;  

    sseq_index a(a_seq); 
    sseq_index b(b_seq); 

    sseq_index_ab ab(a, b); 

    //std::cout << "A" << std::endl << a.desc(1000) << std::endl ; 
    //std::cout << "B" << std::endl << b.desc(1000) << std::endl ; 
   
    std::cout << "AB" << std::endl << ab.desc("BRIEF") << std::endl ; 
    std::cout << "AB" << std::endl << ab.desc("AZERO") << std::endl ; 
    std::cout << "AB" << std::endl << ab.desc("BZERO") << std::endl ; 
    std::cout << "AB" << std::endl << ab.desc("DEVIANT") << std::endl ; 
    //std::cout << "AB" << std::endl << ab.desc("C2INCL") << std::endl ; 

    ab.chi2.save("$FOLD"); 

    return 0 ; 
}
// ~/opticks/sysrap/tests/sseq_index_test.sh 



