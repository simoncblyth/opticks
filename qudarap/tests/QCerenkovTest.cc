#include "Opticks.hh"
#include "SPath.hh"
#include "NP.hh"
#include "QCerenkov.hh"
#include "scuda.h"
#include "OPTICKS_LOG.hh"


void test_check(QCerenkov& ck)
{
    ck.check(); 
}

void test_lookup(QCerenkov& sc)
{
    NP* dst = sc.lookup(); 
    const char* fold = "$TMP/QCerenkovTest" ; 
    LOG(info) << " save to " << fold ; 
    dst->save(fold, "dst.npy"); 
    sc.src->save(fold, "src.npy") ; 
}

void test_GetAverageNumberOfPhotons_s2(const QCerenkov& ck)
{
    LOG(info) ; 
    const double charge = 1. ; 
    double numPhotons, emin, emax ; 
     
    unsigned ni = 1001u ; 
    NP* bis = NP::Linspace<double>(1., 2., ni ); 
    const double* bis_v = bis->cvalues<double>(); 

    NP* scan = NP::Make<double>(ni, 4);   
    double* ss = scan->values<double>();  

    for(unsigned i=0 ; i < bis->shape[0] ; i++ )
    {
        const double BetaInverse = bis_v[i] ; 
        ck.GetAverageNumberOfPhotons_s2<double>(numPhotons, emin, emax, BetaInverse, charge ); 

        ss[4*i+0] = BetaInverse ; 
        ss[4*i+1] = numPhotons ; 
        ss[4*i+2] = emin ; 
        ss[4*i+3] = emax ; 

        if(numPhotons != 0.) 
            std::cout  
                << " i " << std::setw(5) << i 
                << " BetaInverse " << std::fixed << std::setw(10) << std::setprecision(4) << BetaInverse
                << " numPhotons " << std::fixed << std::setw(10) << std::setprecision(4) << numPhotons
                << " emin " << std::fixed << std::setw(10) << std::setprecision(4) << emin
                << " emax " << std::fixed << std::setw(10) << std::setprecision(4) << emax
                << std::endl 
                ; 
    }

    const char* path = SPath::Resolve("$TMP/QCerenkovTest/test_GetAverageNumberOfPhotons_s2.npy"); 
    LOG(info) << " save to " << path ; 
    scan->save(path); 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv); 
    ok.configure(); 

    QCerenkov ck ;  

    //test_lookup(ck); 
    //test_check(ck); 
    test_GetAverageNumberOfPhotons_s2(ck); 

    return 0 ; 
}

