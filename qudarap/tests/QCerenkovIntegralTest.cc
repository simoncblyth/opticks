/**
QCerenkovIntegralTest
======================

Performs many RINDEX s2 integrals for many BetaInverse values
in order to construct an ICDF that is saved into $TMP/QCerenkovIntegralTest

**/

#include "SPath.hh"
#include "NP.hh"
#include "QCerenkovIntegral.hh"
#include "QCK.hh"

#include "scuda.h"
#include "OPTICKS_LOG.hh"


/**
test_GetAverageNumberOfPhotons_s2
-----------------------------------

See ana/ckn.py:compareOtherScans for check that these
results match those from the python prototype and cks 

**/

const char* BASE = "$TMP/QCerenkovIntegralTest" ; 


void test_GetAverageNumberOfPhotons_s2(const QCerenkovIntegral& ck)
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

    int create_dirs = 1 ; //1:filepath
    const char* path = SPath::Resolve(BASE, "test_GetAverageNumberOfPhotons_s2.npy", create_dirs ); 
    LOG(info) << " save to " << path ; 
    scan->save(path); 
}


void test_getS2Integral_UpperCut_one( const QCerenkovIntegral& ck, double BetaInverse )
{
    unsigned nx = 100 ; 
    NP* s2c = ck.getS2Integral_UpperCut(BetaInverse, nx); 

    LOG(info) 
        << " BetaInverse " <<  std::fixed << std::setw(10) << std::setprecision(4) << BetaInverse
        << " s2c " << s2c->desc()
        ; 

  
    int create_dirs = 1 ; // 1:filepath 
    const char* path = SPath::Resolve(BASE, "test_getS2Integral_UpperCut_one.npy", create_dirs ); 
    LOG(info) << " save to " << path ; 
    s2c->save(path); 
}




void test_getS2Integral_UpperCut(const QCerenkovIntegral& ck )
{
    const NP* bis  = NP::Linspace<double>( 1., 2. , 1001 );     // BetaInverse
    unsigned nx = 100u ; 
    NP* s2c = ck.getS2Integral_UpperCut<double>(bis, nx ); 

 
    LOG(info) 
        << std::endl
        << " bis    " << bis->desc()
        << std::endl
        << " s2c " << s2c->desc()
        ; 


    int create_dirs = 2 ; // 2:dirpath 
    const char* fold = SPath::Resolve(BASE, "test_getS2Integral_UpperCut", create_dirs ); 
    LOG(info) << " save to " << fold ; 
    s2c->save(fold,"s2c.npy"); 

    s2c->divide_by_last<double>();  
    s2c->save(fold,"s2cn.npy"); 
}


void test_getS2Integral_SplitBin( const QCerenkovIntegral& ck, const char* bis_, unsigned mul, bool dump )
{
    LOG(info) << "[" ; 
    const NP* bis  = bis_ == nullptr ? NP::Linspace<double>( 1., 2. , 1001 ) : NP::FromString<double>(bis_);     
    NP* s2c = ck.getS2Integral_SplitBin<double>(bis, mul, dump); 

    int create_dirs = 2 ; // 2:dirpath 
    const char* fold = SPath::Resolve(BASE, "test_getS2Integral_SplitBin", create_dirs); 

    LOG(info) 
        << " bis_ " << bis_
        << " mul " << mul 
        << " dump " << dump
        << " s2c " << s2c->sstr() 
        << std::endl 
        << " fold " << fold 
        ; 

    bis->save(fold, "bis.npy"); 
    s2c->save(fold, "s2c.npy"); 
    s2c->divide_by_last<double>();  
    s2c->save(fold, "s2cn.npy"); 
    LOG(info) << "]" ; 
}


void test_makeICDF_UpperCut(const QCerenkovIntegral& ck, unsigned ny, unsigned nx, bool dump )
{
    QCK<double> qck = ck.makeICDF_UpperCut<double>( ny, nx, dump ); 

    LOG(info)
        << std::endl  
        << " qck.bis  " << qck.bis->desc()
        << std::endl  
        << " qck.s2c  " << qck.s2c->desc() 
        << std::endl  
        << " qck.s2cn " << qck.s2cn->desc()
        << std::endl  
        ;

    int create_dirs = 2 ; // 2:dirpath 
    const char* qck_path = SPath::Resolve(BASE, "test_makeICDF_UpperCut",  create_dirs); 
    qck.save(qck_path); 
}



void test_makeICDF_SplitBin(const QCerenkovIntegral& ck, unsigned ny, unsigned mul, bool dump )
{
    QCK<double> qck = ck.makeICDF_SplitBin<double>( ny, mul, dump ); 

    LOG(info)
        << std::endl  
        << " qck.bis  " << qck.bis->desc()
        << std::endl  
        << " qck.s2c  " << qck.s2c->desc() 
        << std::endl  
        << " qck.s2cn " << qck.s2cn->desc()
        << std::endl  
        ;

    int create_dirs = 2 ;  //2:dirpath
    const char* qck_path = SPath::Resolve(BASE, "test_makeICDF_SplitBin",  create_dirs); 
    qck.save(qck_path); 
}




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    char d = 'S' ; 
    char t = argc > 1 ? argv[1][0] : d  ; 
    LOG(info) << " t " << t ; 

    QCerenkovIntegral ck ;  

    if( t == 'N' )
    {
        test_GetAverageNumberOfPhotons_s2(ck); 
    }
    else if( t == 'A' )
    {
        const char* bis = "1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.792" ; 
        //const char* bis = "1.0" ; 
        unsigned mul = 10 ; 
        bool dump = true ; 
        test_getS2Integral_SplitBin(ck, bis, mul, dump) ; 
    }
    else if ( t == 'U' )
    {
        unsigned ny = 2000u ; 
        unsigned nx = 2000u ; 
        bool dump = true ; 
        test_makeICDF_UpperCut(ck, ny, nx, dump ); 
    }
    else if ( t == 'S' )
    {
        unsigned ny = 1000u ; 
        unsigned mul = 100u ; 
        bool dump = true ; 
        test_makeICDF_SplitBin(ck, ny, mul, dump ); 
    }
    return 0 ; 
}

