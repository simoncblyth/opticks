
#include "OPTICKS_LOG.hh"
#include "spath.h"
#include "scuda.h"

#include "NP.hh"
#include "QBase.hh"
#include "QCerenkov.hh"

const char* BASE = "$TMP/QCerenkovTest" ; 

void test_check(QCerenkov& ck)
{
    ck.check(); 
}

void test_lookup(QCerenkov& ck)
{
    NP* icdf_dst = ck.lookup(); 
    if( icdf_dst == nullptr ) return ; 

    const char* fold = spath::Resolve("$TMP/QCerenkovTest/test_lookup") ; 
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

    LOG(info) << "[ QCerenkov " ; 

    QBase qb ;
    LOG(info) << " qb.desc " << qb.desc() ;  


    QCerenkov ck ;  
    LOG(info) << "] QCerenkov " ; 

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

