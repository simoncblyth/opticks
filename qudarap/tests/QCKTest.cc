#include <iostream>
#include <iomanip>
#include <random>

#include "SStr.hh"
#include "SRng.hh"
#include "SPath.hh"
#include "QCK.hh"
#include "NP.hh"
#include "OPTICKS_LOG.hh"

void test_energy_lookup_one( const QCK<double>* qck, double BetaInverse, double u  )
{
    double en = qck->energy_lookup_( BetaInverse, u ); 
    int p = 7 ; 
    int w = p + 10 ; 
    LOG(info)
        << " BetaInverse " << std::fixed << std::setw(w) << std::setprecision(p) << BetaInverse 
        << " u " << std::fixed << std::setw(w) << std::setprecision(p) << u 
        << " en " << std::fixed << std::setw(w) << std::setprecision(p) << en
        ;
}

void test_energy_lookup_many( const QCK<double>* qck, double BetaInverse, unsigned ni )
{
    bool biok = qck->is_permissable(BetaInverse); 
    if(biok == false)
    {
        LOG(fatal) 
            << " BetaInverse not permitted as no photons " << std::fixed << std::setw(10) << std::setprecision(4) << BetaInverse
            ; 
        return ; 
    }

    NP* uu = NP::MakeUniform<double>( ni ) ; 
    NP* en = qck->energy_lookup(BetaInverse, uu ) ; 
    const char* path = SPath::MakePath<double>( "$TMP/QCKTest", nullptr, BetaInverse, "test_energy_lookup_many.npy" ); 
    en->save(path);     
}


void test_energy_sample_one( const QCK<double>* qck, double BetaInverse )
{
    unsigned seed = 0u ; 
    SRng<double> rng(seed) ;  

    double en = qck->energy_sample_( BetaInverse, rng ); 

    int p = 7 ; 
    int w = p + 10 ; 

    LOG(info)
        << " BetaInverse " << std::fixed << std::setw(w) << std::setprecision(p) << BetaInverse 
        << " en " << std::fixed << std::setw(w) << std::setprecision(p) << en
        ;
}

void test_energy_sample_many( const QCK<double>* qck, double BetaInverse, unsigned ni )
{
    bool biok = qck->is_permissable(BetaInverse); 
    if(biok == false)
    {
        LOG(fatal) 
            << " BetaInverse not permitted as no photons " << std::fixed << std::setw(10) << std::setprecision(4) << BetaInverse
            ; 
        return ; 
    }

    unsigned seed = 0u ; 
    SRng<double> rng(seed) ;  
    
    NP* en = qck->energy_sample( BetaInverse, rng, ni ) ; 
    const char* path = SPath::MakePath<double>( "$TMP/QCKTest", nullptr, BetaInverse, "test_energy_sample_many.npy" ); 
    en->save(path) ;     
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* icdf_path = SPath::Resolve("$TMP/QCerenkovTest/test_makeICDF"); 
    QCK<double>* qck = QCK<double>::Load(icdf_path); 
    qck->init(); 

    LOG(info) << " qck.bis  " << qck->bis->desc(); 
    LOG(info) << " qck.s2c  " << qck->s2c->desc(); 
    LOG(info) << " qck.s2cn " << qck->s2cn->desc(); 

    unsigned ni = 1000000 ; 

    std::vector<double> bis ;  
    for( double bi=1.0 ; bi < qck->rmx ; bi+=0.05 ) bis.push_back(bi); 

    //bis.push_back(1.75);   // about mean of 1 photon 
    //bis.push_back(1.792);  // extreme peak : some tiny fraction of a photon  
    //bis.push_back(1.45);   // pdomain assert, from going slightly non-monotonic


    for(unsigned i=0 ; i < bis.size() ; i++)
    {
        double BetaInverse = bis[i] ; 
        LOG(info) 
            << " BetaInverse " << std::fixed << std::setw(10) << std::setprecision(4) << BetaInverse
            ; 
 
        test_energy_lookup_many( qck, BetaInverse, ni ); 
        test_energy_sample_many( qck, BetaInverse, ni ); 
    }

    return 0 ; 
}

