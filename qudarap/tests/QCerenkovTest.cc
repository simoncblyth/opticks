
#include "SPath.hh"
#include "NP.hh"
#include "QCerenkov.hh"

#include "scuda.h"
#include "OPTICKS_LOG.hh"

const char* BASE = "$TMP/QCerenkovTest" ; 


void test_check(QCerenkov& ck)
{
    ck.check(); 
}

void test_lookup(QCerenkov& ck)
{
    NP* icdf_dst = ck.lookup(); 
    if( icdf_dst == nullptr ) return ; 

    int create_dirs = 2 ; // 2:dirpath
    const char* fold = SPath::Resolve("$TMP/QCerenkovTest", "test_lookup", create_dirs) ; 
    LOG(info) << " save to " << fold ; 

    icdf_dst->save(fold, "icdf_dst.npy"); 
    ck.icdf->save(fold,  "icdf_src.npy") ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    char d = 'L' ; 
    char t = argc > 1 ? argv[1][0] : d  ; 
    LOG(info) << " t " << t ; 

    QCerenkov ck ;  

    if( t == 'L' )
    {
        test_lookup(ck); 
    }
    else if( t == 'C' )
    {
        test_check(ck); 
    }


    return 0 ; 
}

