
#include "SPath.hh"
#include "NP.hh"
#include "QCerenkov.hh"
#include "QCK.hh"

#include "scuda.h"
#include "OPTICKS_LOG.hh"


void test_check(QCerenkov& ck)
{
    ck.check(); 
}

void test_lookup(QCerenkov& ck)
{
    NP* dst = ck.lookup(); 
    const char* fold = "$TMP/QCerenkovTest" ; 
    LOG(info) << " save to " << fold ; 
    dst->save(fold, "dst.npy"); 
    ck.src->save(fold, "src.npy") ; 
}

/**
test_GetAverageNumberOfPhotons_s2
-----------------------------------

See ana/ckn.py:compareOtherScans for check that these
results match those from the python prototype and cks 

**/

void test_GetAverageNumberOfPhotons_s2(const QCerenkov& ck)
{
    LOG(info) ; 
    const double charge = 1. ; 
    double numPhotons, emin, emax ; 
     
    unsigned ni = 1001u ; 
    NP* bis = NP::Linspace<double>(1., 2., ni ); 
    const double* bb = bis->cvalues<double>(); 

    NP* scan = NP::Make<double>(ni, 4);   
    double* ss = scan->values<double>();  

    for(unsigned i=0 ; i < unsigned(bis->shape[0]) ; i++ )
    {
        const double BetaInverse = bb[i] ; 
        numPhotons = ck.GetAverageNumberOfPhotons_s2<double>(emin, emax, BetaInverse, charge ); 

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


void test_getS2CumulativeIntegrals_one( const QCerenkov& ck, double BetaInverse )
{
    unsigned nx = 100 ; 
    NP* s2c = ck.getS2CumulativeIntegrals(BetaInverse, nx); 

    LOG(info) 
        << " BetaInverse " <<  std::fixed << std::setw(10) << std::setprecision(4) << BetaInverse
        << " s2c " << s2c->desc()
        ; 

    const char* path = SPath::Resolve("$TMP/QCerenkovTest/test_getS2CumulativeIntegrals_one.npy"); 
    LOG(info) << " save to " << path ; 
    s2c->save(path); 
}


void test_getS2CumulativeIntegrals_many(const QCerenkov& ck )
{
    const NP* bis  = NP::Linspace<double>( 1., 2. , 1001 );     // BetaInverse
    unsigned nx = 100u ; 
    NP* s2c = ck.getS2CumulativeIntegrals<double>(bis, nx ); 

 
    LOG(info) 
        << std::endl
        << " bis    " << bis->desc()
        << std::endl
        << " s2c " << s2c->desc()
        ; 

    const char* s2c_path = SPath::Resolve("$TMP/QCerenkovTest/test_getS2CumulativeIntegrals_many_s2c.npy"); 
    LOG(info) << " save to " << s2c_path ; 
    s2c->save(s2c_path); 

    const char* s2cn_path = SPath::Resolve("$TMP/QCerenkovTest/test_getS2CumulativeIntegrals_many_s2cn.npy"); 
    s2c->divide_by_last<double>();  
    s2c->save(s2cn_path); 
}


void test_makeICDF(const QCerenkov& ck )
{
    unsigned ny = 1000u ; 
    unsigned nx = 100u ; 

    QCK icdf = ck.makeICDF<double>( ny, nx ); 

    LOG(info)
        << std::endl  
        << " icdf.bis  " << icdf.bis->desc()
        << std::endl  
        << " icdf.s2c  " << icdf.s2c->desc() 
        << std::endl  
        << " icdf.s2cn " << icdf.s2cn->desc()
        << std::endl  
        ;
}



void test_getS2SliverIntegrals_one(const QCerenkov& ck, double BetaInverse )
{
    double emin, emax ; 
    const NP* edom = NP::Linspace<double>( 1.55, 15.5, 280 ); // steps of 0.05  
    NP* s2slv = ck.getS2SliverIntegrals(emin, emax, BetaInverse, edom ); 

    LOG(info) 
        << " BetaInverse " <<  std::fixed << std::setw(10) << std::setprecision(4) << BetaInverse
        << " emin " <<  std::fixed << std::setw(10) << std::setprecision(4) << emin
        << " emax " <<  std::fixed << std::setw(10) << std::setprecision(4) << emax
        << " s2slv " << s2slv->desc()
        ; 

    const char* path = SPath::Resolve("$TMP/QCerenkovTest/test_getS2SliverIntegrals_one.npy"); 
    LOG(info) << " save to " << path ; 
    s2slv->save(path); 

}


void test_getS2SliverIntegrals_many(const QCerenkov& ck )
{
    const NP* bis  = NP::Linspace<double>( 1., 2. , 1001 );     // BetaInverse
    const NP* edom = NP::Linspace<double>( 1.55, 15.5, 280 );   // steps of 0.05 eV

    NP* s2slv = ck.getS2SliverIntegrals<double>(bis, edom ); 
    NP* cs2slv = s2slv->cumsum<double>(1u); 

    LOG(info) 
        << std::endl
        << " bis    " << bis->desc()
        << std::endl
        << " edom   " << edom->desc()
        << std::endl
        << " s2slv  " << s2slv->desc()
        << std::endl
        << " cs2slv " << cs2slv->desc()
        ; 

    const char* s2slv_path = SPath::Resolve("$TMP/QCerenkovTest/test_getS2SliverIntegrals_many.npy"); 
    LOG(info) << " save to " << s2slv_path ; 
    s2slv->save(s2slv_path); 

    const char* cs2slv_path = SPath::Resolve("$TMP/QCerenkovTest/test_getS2SliverIntegrals_many_cs2slv.npy"); 
    LOG(info) << " save to " << cs2slv_path ; 
    cs2slv->save(cs2slv_path); 

    const char* cs2slv_norm_path = SPath::Resolve("$TMP/QCerenkovTest/test_getS2SliverIntegrals_many_cs2slv_norm.npy"); 
    cs2slv->divide_by_last<double>();  
    cs2slv->save(cs2slv_norm_path); 


}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    QCerenkov ck ;  
    //const double BetaInverse = 1.5 ; 
    //const double BetaInverse = 1.0 ; 

    //test_lookup(ck); 
    //test_check(ck); 

    //test_GetAverageNumberOfPhotons_s2(ck); 
    //test_getS2SliverIntegrals_one(ck,BetaInverse) ; 
    //test_getS2SliverIntegrals_many(ck) ; 

    //test_getS2CumulativeIntegrals_one(ck,BetaInverse) ; 
    //test_getS2CumulativeIntegrals_many(ck) ; 

    test_makeICDF(ck); 


    return 0 ; 
}

