#pragma once
/**
sstampfold.h
===============

Presents sstamp.h timestamp metadata from a collection of SEvt NPFold

**/

#include "NPFold.h"

struct sstampfold
{
    static constexpr const char* COUNT_KEY = "subcount" ; 
    static constexpr const char* STAMP_KEY = "substamp" ; 
    static constexpr const char* DELTA_KEY = "delta_substamp" ; 
    static constexpr const char* LABELS_KEY = "labels" ; 

    const NPFold* af ; 
    const char* symbol ; 

    const NP* sc ;  
    const NP* st ;  
    const NP* dt ;
    const NP* la ;

    const std::vector<std::string>* rows ; 
    const std::vector<std::string>* cols ; 
 
    sstampfold( const NPFold* af, const char* symbol ); 
    std::string brief() const ; 
    std::string desc() const ; 

    static NP* BOA(const sstampfold& a, const sstampfold& b); 

};


inline sstampfold::sstampfold( const NPFold* af_, const char* symbol_ ) 
    :
    af(af_),
    symbol(symbol_ ? strdup(symbol_) : nullptr),
    sc(af ? af->get(COUNT_KEY) : nullptr),
    st(af ? af->get(STAMP_KEY) : nullptr),
    dt(af ? af->get(DELTA_KEY) : nullptr),
    la(af ? af->get(LABELS_KEY) : nullptr),
    rows(st ? &(st->names) : nullptr), 
    cols(la ? &(la->names) : nullptr)
{
}
inline std::string sstampfold::brief() const 
{
    std::stringstream ss ; 
    ss << "[sstampfold::brief" << std::endl 
       << " af " << ( af ? "YES" : "NO " ) << std::endl 
       << " symbol " << ( symbol ? symbol : "-" ) << std::endl  
       << " sc " << ( sc ? sc->sstr() : "-" ) << std::endl 
       << " st " << ( st ? st->sstr() : "-" ) << std::endl 
       << " dt " << ( dt ? dt->sstr() : "-" ) << std::endl 
       << " la " << ( la ? la->sstr() : "-" ) << std::endl 
       << "]sstampfold::brief" << std::endl 
       ; 
    std::string str =  ss.str() ; 
    return str ; 
}

inline std::string sstampfold::desc() const 
{
    std::stringstream ss ; 
    //ss << brief() << std::endl ; 

    ss 
       << "[sstampfold::desc " << ( symbol ? symbol : "-" )  << std::endl 
       << ( sc ? sc->descTable<int>(8) : "-" ) << std::endl 
       << std::endl 
       << ( dt ? dt->descTable_<int64_t>(8,cols,rows) : "-" ) << std::endl 
       << "]sstampfold::desc " << ( symbol ? symbol : "-" )  << std::endl 
       ;
    std::string str =  ss.str() ; 
    return str ; 
} 

/**
sstampfold::BOA
-----------------

B over A

**/

inline NP* sstampfold::BOA( const sstampfold& a, const sstampfold& b)  // static
{
    const NP* adt = a.dt ; 
    const NP* bdt = b.dt ;

    

    std::cout 
        << "[sstampfold::BOA" 
        << std::endl     
        << " adt.shape " 
        << ( adt ? adt->sstr() : "-" ) 
        << std::endl     
        << " bdt.shape " 
        << ( bdt ? bdt->sstr() : "-" ) 
        << std::endl     
        ;

    assert( adt && adt->shape.size() == 2 ); 
    assert( bdt && bdt->shape.size() == 2 ); 


    int a_ni = adt->shape[0] ;  
    int b_ni = bdt->shape[0] ;  

    //assert( int(adt->names.size()) == a_ni ); 
    //assert( int(bdt->names.size()) == b_ni ); 

    std::cout << " adt->names.size " << adt->names.size() << std::endl ; 
    std::cout << " bdt->names.size " << bdt->names.size() << std::endl ; 

    std::cout << " adt->labels " << ( adt->labels ? "YES" : "NO " ) << std::endl ; 
    std::cout << " bdt->labels " << ( bdt->labels ? "YES" : "NO " ) << std::endl ; 

    assert( a_ni == b_ni ); 
    int ni = a_ni ; 


    int a_nj = adt->shape[1] ; 
    int b_nj = bdt->shape[1] ; 

    const int64_t* aa = adt->cvalues<int64_t>();  
    const int64_t* bb = bdt->cvalues<int64_t>();  

    int c_ni = ni ; 
    int c_nj = 3 ; 
    NP* c = NP::Make<double>(c_ni, c_nj); 
    double* cc = c->values<double>(); 

    for(int i=0 ; i < ni ; i++)
    {
        int64_t av = aa[i*a_nj+a_nj-1] ; 
        int64_t bv = bb[i*b_nj+b_nj-1] ; 
        double  boa = double(bv)/double(av); 

        cc[i*c_nj+0] = av ;   
        cc[i*c_nj+1] = bv ;   
        cc[i*c_nj+2] = boa ;   

        std::cout 
            //<< std::setw(10) << adt->names[i] 
            << std::setw(10) << av 
            //<< std::setw(10) << bdt->names[i] 
            << std::setw(10) << bv
            << std::fixed << std::setw(10) << std::setprecision(2) << boa
            << std::endl 
            ;
    }
    return c ; 
}







    


