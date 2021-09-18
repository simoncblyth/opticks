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
    double dt ; 
    double en = qck->energy_lookup_( BetaInverse, u, dt ); 
    int p = 7 ; 
    int w = p + 10 ; 
    LOG(info)
        << " BetaInverse " << std::fixed << std::setw(w) << std::setprecision(p) << BetaInverse 
        << " u " << std::fixed << std::setw(w) << std::setprecision(p) << u 
        << " en " << std::fixed << std::setw(w) << std::setprecision(p) << en
        << " dt " << std::fixed << std::setw(w) << std::setprecision(p) << dt
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

    NP* tt = NP::Make<double>( ni ); 
    NP* uu = NP::MakeUniform<double>( ni ) ; 
    NP* en = qck->energy_lookup(BetaInverse, uu, tt ) ; 
    const char* path = SPath::MakePath<double>( "$TMP/QCKTest", nullptr, BetaInverse, "test_energy_lookup_many.npy" ); 
    en->save(path);     

    const char* tt_path = SPath::MakePath<double>( "$TMP/QCKTest", nullptr, BetaInverse, "test_energy_lookup_many_tt.npy" ); 
    tt->save(tt_path);     
}


void test_energy_sample_one( const QCK<double>* qck, double BetaInverse )
{
    double dt ; 
    unsigned seed = 0u ; 
    SRng<double> rng(seed) ;  

    unsigned count ; 
    double en = qck->energy_sample_( BetaInverse, rng, dt, count ); 

    int p = 7 ; 
    int w = p + 10 ; 

    LOG(info)
        << " BetaInverse " << std::fixed << std::setw(w) << std::setprecision(p) << BetaInverse 
        << " en " << std::fixed << std::setw(w) << std::setprecision(p) << en
        << " dt " << std::fixed << std::setw(w) << std::setprecision(p) << dt
        << " count " << std::setw(7) << count 
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
    
    NP* tt = NP::Make<double>( ni ); 
    NP* en = qck->energy_sample( BetaInverse, rng, ni, tt ) ; 
    const char* path = SPath::MakePath<double>( "$TMP/QCKTest", nullptr, BetaInverse, "test_energy_sample_many.npy" ); 
    en->save(path) ;     

    const char* tt_path = SPath::MakePath<double>( "$TMP/QCKTest", nullptr, BetaInverse, "test_energy_sample_many_tt.npy" ); 
    tt->save(tt_path) ;     
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


    std::vector<double> bis ;  
    for( double bi=1.0 ; bi < qck->rmx ; bi+=0.05 ) bis.push_back(bi); 
    bis.push_back(1.792);  // extreme peak : some tiny fraction of a photon  
    //bis.push_back(1.45);   // pdomain assert, from going slightly non-monotonic


    double emin, emax ; 
    for(unsigned i=0 ; i < bis.size() ; i++)
    {
        double BetaInverse = bis[i] ; 
        qck->energy_range( emin, emax, BetaInverse, true ); 
    }

    for(unsigned i=0 ; i < bis.size() ; i++)
    {
        unsigned num_gen = 1000000 ; 
        double BetaInverse = bis[i] ; 
        LOG(info) << " BetaInverse " << std::fixed << std::setw(10) << std::setprecision(4) << BetaInverse ;
        test_energy_lookup_many( qck, BetaInverse, num_gen ); 
        test_energy_sample_many( qck, BetaInverse, num_gen ); 
    }

    return 0 ; 
}

