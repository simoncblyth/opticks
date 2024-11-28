#pragma once
/**
sseq_index.h 
===============

This reimplements some python/NumPy that is slow with large seq.npy

Basis struct from sseq.h and NPX.h
-------------------------------------

sseq
   sequence flag and boundary history "seqhis" "seqbnd" of a single photon, 
   up to 32 step points 

NPX.h
    NPX::VecFromArray<sseq> reads vector<sseq> from NP seq array
   

Following structs are defined
-------------------------------

sseq_index_count
   struct holding index and count  

sseq_index_count_ab
   struct holding a and b sseq_index_count to facilitate comparisons

sseq_unique
   struct holding q:sseq and ic:sseq_index_count
   
sseq_qab   
   struct holding q:sseq a:sseq_index_count b:sseq_index_count

sseq_index
   reimplementation of ~/opticks/ana/qcf.py:QU

   * q:vector of sseq (populated from typically large input array within ctor)
   * m:map of unique sseq with counts and first indices (relies on sseq.h hash specialization)
   * u:descending count ordered vector of sseq_unique 


Q: reimplementation of ~/opticks/ana/qcf.py:QCF ?

**/

#include "ssys.h"
#include "sseq.h"
#include "NPX.h"


struct sseq_index_count
{
    int index ; 
    int count ; 
    std::string desc() const ; 
};
inline std::string sseq_index_count::desc() const
{
    std::stringstream ss ;  
    ss << std::setw(7) << count << " : " << std::setw(7) << index  ;  
    std::string str = ss.str(); 
    return str ; 
}

struct sseq_index_count_ab
{
    sseq_index_count a ; 
    sseq_index_count b ; 
};


struct sseq_unique
{
    sseq             q  ; 
    sseq_index_count ic ; 

    bool operator< (const sseq_unique& other) const { return ic.count < other.ic.count ; }
    std::string desc() const ; 
}; 

inline std::string sseq_unique::desc() const
{
    std::stringstream ss ;  
    ss << q.seqhis_() << " : " << ic.desc() ; 
    std::string str = ss.str(); 
    return str ; 
} 



struct sseq_qab 
{
    sseq             q ; 
    sseq_index_count a ; 
    sseq_index_count b ; 

    double c2(bool& included, int absum_min) const ;

    int maxcount() const { return std::max(a.count, b.count) ; } 
    bool operator< (const sseq_qab& other) const { return maxcount() < other.maxcount() ; } 

    std::string desc(int absum_min) const ; 
}; 

inline double sseq_qab::c2(bool& included, int absum_min) const 
{
    double _a = a.count ; 
    double _b = b.count ; 
    double absum = _a + _b ;
    double abdif = _a - _b ; 
    included = _a > 0 && _b > 0 && absum > absum_min ;  
    return included ? abdif*abdif/absum : 0 ;   
}


inline std::string sseq_qab::desc(int absum_min) const
{
    bool included(false); 

    std::stringstream ss ;  
    ss << q.seqhis_() 
       << " : " 
       << std::setw(7) << a.count 
       << std::setw(7) << b.count 
       << " : "
       << std::fixed << std::setw(10) << std::setprecision(4) << c2(included, absum_min) 
       << " : "
       << ( included ? "Y" : "N" ) 
       << " : "
       << std::setw(7) << a.index 
       << std::setw(7) << b.index 
       ;  
    std::string str = ss.str(); 
    return str ; 
} 



struct sseq_index
{
    std::vector<sseq> q ;                   // typically large input array 

    std::map<sseq, sseq_index_count> m ;    // map of unique sseq with counts and first indices
    std::vector<sseq_unique> u ;            // descending count ordered vector of sseq_unique 

    sseq_index( const NP* seq); 

    void load_seq( const NP* seq ); 
    void count_unique(); 
    void order_seq();
    std::string desc(int min_count=0) const; 
}; 


inline sseq_index::sseq_index( const NP* seq)
{
    load_seq(seq); 
    count_unique(); 
    order_seq(); 
}


inline void sseq_index::load_seq(const NP* seq)
{
    NPX::VecFromArray<sseq>(q, seq ); 
}

/**
sseq_index::count_unique fill the sseq keyed map of occurence counts
----------------------------------------------------------------------

Iterate over the source vector populating the 
map with the index of first occurrence and
count of the frequencey of occurrence.  

Relies on sseq hash specialization based on seqhis values

**/

inline void sseq_index::count_unique()
{ 
    for (int i = 0; i < int(q.size()); i++) 
    {
        const sseq& seq = q[i];

        std::map<sseq, sseq_index_count>::iterator it  = m.find(seq);

        if(it == m.end()) 
        {
            int q_index_of_first_occurrence = i ;  
            m[seq] = {q_index_of_first_occurrence, 1} ;
        }
        else 
        {
            it->second.count++ ; 
        }
    }
}

/**
sseq_index::order_seq : sorting unique sseq in descending count order
----------------------------------------------------------------------

1. copy from map m into vector u 
2. sort the u vector into descending count order

**/


inline void sseq_index::order_seq()
{
    for(auto it=m.begin() ; it != m.end() ; it++) u.push_back( { it->first, {it->second.index, it->second.count} } );  

    auto descending_order = [](const sseq_unique& a, const sseq_unique& b) { return a.ic.count > b.ic.count ; } ; 

    std::sort( u.begin(), u.end(), descending_order  ); 
}

inline std::string sseq_index::desc(int min_count) const
{
    std::stringstream ss ; 
    int num = u.size(); 
    ss << "[sseq_index::desc num " << num << std::endl ; 
    for(int i=0 ; i < num ; i++) 
    {
        if( u[i].ic.count < min_count ) break ; 
        ss << u[i].desc() << std::endl ; 
    }
    ss << "]sseq_index::desc" << std::endl ; 
    std::string str = ss.str() ; 
    return str ; 
}


struct sseq_index_ab_chi2
{
    static constexpr const char* NAME = "sseq_index_ab_chi2.npy" ; 
    static const int DEFAULT_ABSUM_MIN = 30 ; 
    static constexpr const char* EKEY = "sseq_index_ab_chi2_ABSUM_MIN" ;  // equiv to python C2CUT

    double sum ; 
    double ndf ; 
    double absum_min ; 
    double spare ; 

    NP* serialize() const ; 
    void save(const char* dir) const ; 

    void init(); 
    std::string desc() const ; 
};


inline NP* sseq_index_ab_chi2::serialize() const 
{
    NP* a = NP::Make<double>(4) ; 
    double* aa = a->values<double>(); 
    aa[0] = sum ; 
    aa[1] = ndf ; 
    aa[2] = absum_min ; 
    aa[3] = spare ; 
    return a ; 
}
inline void sseq_index_ab_chi2::save(const char* dir) const 
{
    NP* a = serialize();
    if(dir == nullptr) a->save(NAME); 
    else a->save(dir, NAME); 
}


inline void sseq_index_ab_chi2::init()
{
    sum = 0 ; 
    ndf = 0 ; 
    absum_min = ssys::getenvint(EKEY,DEFAULT_ABSUM_MIN) ; 
    spare = 0 ; 
}

std::string sseq_index_ab_chi2::desc() const 
{
    std::stringstream ss ; 
    ss << "sseq_index_ab_chi2::desc"
       << " sum " << std::fixed << std::setw(10) << std::setprecision(4) << sum
       << " ndf " << std::setw(7) << ndf  
       << " sum/ndf " << std::fixed << std::setw(10) << std::setprecision(4) << sum/ndf 
       << " " << EKEY 
       << ":" 
       << absum_min 
       ;
    std::string str = ss.str() ; 
    return str ; 
}


/**
sseq_index_ab


**/

struct sseq_index_ab
{
    const sseq_index& a ; 
    const sseq_index& b ;

    std::map<sseq, sseq_index_count_ab> m ;  
    std::vector<sseq_qab> u ;
    sseq_index_ab_chi2    chi2 ;  

    sseq_index_ab( const sseq_index& a_, const sseq_index& b_ ); 

    void init();  
    void collect_seq(); 
    void order_seq(); 
    void calc_chi2(); 

    std::string desc(const char* opt) const ; 

};

inline sseq_index_ab::sseq_index_ab( const sseq_index& a_, const sseq_index& b_ )
    :
    a(a_),
    b(b_)
{
    init(); 
}

inline void sseq_index_ab::init()
{
    collect_seq();  
    order_seq(); 
    calc_chi2(); 
}

/**
sseq_index_ab::collect_seq
---------------------------

sseq_index_count_ab 
     4 integers with index and count from A and B


**/


inline void sseq_index_ab::collect_seq()
{
    // collect from A into the map 
    for (int i = 0; i < int(a.u.size()); i++) 
    {
        const sseq_unique& q_ic = a.u[i];
        const sseq& q = q_ic.q ;
        std::map<sseq, sseq_index_count_ab>::iterator it  = m.find(q);
        bool first_q = it == m.end() ;               // first find of sseq history q within m  
        if(first_q) m[q] = { q_ic.ic , {-1,-1} } ;   // fill in the two A slots setting -1,-1 for B 
    }
    // should always be first_q as m starts empty and are grabbing from uniques already 


    // collect from B into the map 
    for (int i = 0; i < int(b.u.size()); i++) 
    {
        const sseq_unique& q_ic = b.u[i];
        const sseq& q = q_ic.q ;
        std::map<sseq, sseq_index_count_ab>::iterator it  = m.find(q_ic.q);

        bool first_q = it == m.end() ; // first find of sseq history q within m  
      
        if(first_q) m[q] = { {-1,-1}, q_ic.ic } ;   // fill in the two B slots setting -1,-1 for A
        else it->second.b = q_ic.ic ; 
        // already found (from A) so just fill in the B slot 

    }
}

inline void sseq_index_ab::order_seq()
{
    // copy from map to vector
    for(auto it=m.begin() ; it != m.end() ; it++) 
    {
        const sseq& q = it->first ; 
        const sseq_index_count_ab& ab = it->second ; 
        u.push_back( {q, ab.a, ab.b } );  
    }

    // order the vector in descending max count order 
    auto descending_order = [](const sseq_qab& x, const sseq_qab& y) { return x.maxcount() > y.maxcount() ; } ; 
    std::sort( u.begin(), u.end(), descending_order ); 
}


inline void sseq_index_ab::calc_chi2()
{
    int num = u.size(); 
    chi2.init(); 
    for(int i=0 ; i < num ; i++)
    {
        const sseq_qab& qab = u[i] ;
        bool included = false ;  
        double c2 = qab.c2(included, chi2.absum_min) ; 
        if(included)
        {
            chi2.sum += c2 ; 
            chi2.ndf +=  1 ; 
        }
    }
}

/**x
sseq_index_ab::desc
---------------------

HMM: inclusion in the chi2 is based on sum of a and b counts

**/

inline std::string sseq_index_ab::desc(const char* opt) const 
{
    enum { ALL, AZERO, BZERO, C2INC, C2EXC, DEVIANT, BRIEF } ; 
    int mode = BRIEF ; 
    if(      strstr(opt,"ALL")  )   mode = ALL ;
    else if( strstr(opt,"AZERO") )  mode = AZERO ; 
    else if( strstr(opt,"BZERO") )  mode = BZERO ; 
    else if( strstr(opt,"C2INC") )  mode = C2INC ; 
    else if( strstr(opt,"C2EXC") )  mode = C2EXC ; 
    else if( strstr(opt,"DEVIANT")) mode = DEVIANT ; 
    else if( strstr(opt,"BRIEF"))   mode = BRIEF ; 

    int num = u.size(); 
    std::stringstream ss ; 
    ss 
        << "[sseq_index_ab::desc u.size " << num  
        << " opt " << opt 
        << " mode " << mode
        << ( mode == BRIEF ? chi2.desc()  : "" ) 
        << std::endl 
        ;
    for(int i=0 ; i < num ; i++)
    {
        const sseq_qab& qab = u[i] ; 

        // hmm : inclusion based on max count OR sum of counts ? 
        //int abx = std::max( qab.a.count, qab.b.count ) ; 
        //if( abx < min_abx ) continue ; 

        bool c2_inc(false); 
        double c2 = qab.c2(c2_inc, chi2.absum_min); 

        bool a_zero = qab.a.count <= 0 && qab.b.count > 10 ; 
        bool b_zero = qab.b.count <= 0 && qab.a.count > 10 ; 
        bool deviant = c2 > 10. ; 

        bool selected = true ; 
        switch(mode)
        {
            case ALL:     selected = true    ; break ; 
            case BRIEF:   selected = i < 60  ; break ; 
            case AZERO:   selected = a_zero  ; break ; 
            case BZERO:   selected = b_zero  ; break ; 
            case C2INC:   selected = c2_inc  ; break ; 
            case C2EXC:   selected = !c2_inc ; break ; 
            case DEVIANT: selected = deviant ; break ; 
        }

        const char* head = deviant ? ":r:`" : "    " ; 
        const char* tail = deviant ? "`" : " " ; 
   

        if(selected) ss 
           << head
           << qab.q.seqhis_() 
           << " : " 
           << std::setw(7) << qab.a.count 
           << std::setw(7) << qab.b.count 
           << " : " 
           << std::fixed << std::setw(10) << std::setprecision(4) << c2 
           << " : " 
           << ( c2_inc ? "Y" : "N" )
           << " : " 
           << std::setw(7) << qab.a.index
           << std::setw(7) << qab.b.index
           << " : "
           << ( a_zero ? "AZERO " : "" )
           << ( b_zero ? "BZERO " : "" )
           << ( deviant ? "DEVIANT " : "" )
           << ( c2_inc ? " " : "C2EXC " )   // SUPPRESS C2INC AS ITS NORMAL
           << tail 
           << std::endl 
           ;
    }
    ss
        << "]sseq_index_ab::desc" << std::endl 
        ;

    std::string str = ss.str(); 
    return str ; 
}


