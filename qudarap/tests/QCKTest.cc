#include <iostream>
#include <iomanip>
#include <random>

#include "SPath.hh"
#include "QCK.hh"
#include "NP.hh"
#include "OPTICKS_LOG.hh"



void test_energy_lookup_one( const QCK* qck, double BetaInverse, double u  )
{
    double en = qck->energy_lookup( BetaInverse, u ); 

    int p = 7 ; 
    int w = p + 10 ; 

    LOG(info)
        << " BetaInverse " << std::fixed << std::setw(w) << std::setprecision(p) << BetaInverse 
        << " u " << std::fixed << std::setw(w) << std::setprecision(p) << u 
        << " en " << std::fixed << std::setw(w) << std::setprecision(p) << en
        ;

}


void test_energy_lookup_many( const QCK* qck )
{
    unsigned ni = 1000000 ; 
    //unsigned ni = 100 ; 
    NP* uu = NP::MakeUniform<double>( ni ) ; 
    
    double BetaInverse = 1.5 ; 
    NP* en = qck->energy_lookup(BetaInverse, uu ) ; 

    const char* path = SPath::Resolve("$TMP/QCKTest/test_energy_lookup_many/en.npy") ; 
    en->save(path);     
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* icdf_path = SPath::Resolve("$TMP/QCerenkovTest/test_makeICDF"); 
    QCK* qck = QCK::Load(icdf_path); 

    LOG(info) << " qck.bis  " << qck->bis->desc(); 
    LOG(info) << " qck.s2c  " << qck->s2c->desc(); 
    LOG(info) << " qck.s2cn " << qck->s2cn->desc(); 

    //double BetaInverse = 1.5 ; 
    //double u = 0.5 ; 
    //test_energy_lookup_one(  qck, BetaInverse, u  ); 

    test_energy_lookup_many( qck ); 


    return 0 ; 
}

