/**
sysrap/tests/sseq_index_test.sh : C++ photon history comparison using seq.npy arrays
=====================================================================================

1. loads arrays a_seq and b_seq from $AFOLD/seq.npy and $BFOLD/seq.npy 
2. instanciate sseq_index a and b from the a_seq and b_seq arrays
3. instanciate sseq_index_ab comparing a and b 
4. report on history differences between a and b using sseq_index_ab methods
5. saves comparison metadata to $FOLD directory 

::

    ~/opticks/sysrap/tests/sseq_index_test.sh 


Requirements
-------------

* AFOLD and BFOLD envvars pointing to SEvt folders 
  containing seq.npy photon history files


**/


#include "NP.hh"
#include "ssys.h"
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

    const char* _DEBUG = "sseq_index_test__DEBUG" ; 
    int DEBUG = ssys::getenvint(_DEBUG, 0);  
    std::cout << _DEBUG << ":" << DEBUG << "\n" ; 

    sseq_index a(a_seq); 
    sseq_index b(b_seq); 

    sseq_index_ab ab(a, b); 

    if( DEBUG > 0 )
    {
        std::cout << "A" << std::endl << a.desc(1000) << std::endl ; 
        std::cout << "B" << std::endl << b.desc(1000) << std::endl ; 
    }
   
    std::cout << "AB" << std::endl << ab.desc("BRIEF") << std::endl ; 
    std::cout << "AB" << std::endl << ab.desc("AZERO") << std::endl ; 
    std::cout << "AB" << std::endl << ab.desc("BZERO") << std::endl ; 
    std::cout << "AB" << std::endl << ab.desc("DEVIANT") << std::endl ; 
    
    if(DEBUG > 0)
    {
        std::cout << "AB" << std::endl << ab.desc("C2INCL") << std::endl ; 
    }


    ab.chi2.save("$FOLD"); 

    return 0 ; 
}
// ~/opticks/sysrap/tests/sseq_index_test.sh 



