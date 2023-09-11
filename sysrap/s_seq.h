#pragma once

#include "NP.hh"
#include "ssys.h"

struct s_seq
{
    static constexpr const char* OPTICKS_RANDOM_SEQPATH = "OPTICKS_RANDOM_SEQPATH" ;
    static constexpr const char* DEFAULT_SEQPATH = 
    "$HOME/.opticks/precooked/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000/rng_sequence_f_ni100000_nj16_nk16_ioffset000000.npy" ;
    static const char* SeqPath() ; 

    s_seq(); 

    std::string desc() const  ; 
    double flat();
    std::string demo(int n) ; 
    int  getSequenceIndex() const ; 
    void setSequenceIndex(int index_);   // -ve to disable, must be less than ni  
    bool is_enabled() const ; 

private:
    const char*   m_seqpath ; 
    const NP*     m_seq ; 
    const float*  m_seq_values ;
    int           m_seq_ni ;
    int           m_seq_nv ;
    int           m_seq_index ;
    NP*           m_cur ;
    int*          m_cur_values ;
    bool          m_recycle ;
    double        m_flat_prior ;
};

inline const char* s_seq::SeqPath() // static
{
    return ssys::getenvvar(OPTICKS_RANDOM_SEQPATH, DEFAULT_SEQPATH) ; 
}

inline s_seq::s_seq()
    :
    m_seqpath(U::Resolve(SeqPath())),
    m_seq(m_seqpath ? NP::Load(m_seqpath) : nullptr),
    m_seq_values(m_seq ? m_seq->cvalues<float>() : nullptr ),
    m_seq_ni(m_seq ? m_seq->shape[0] : 0 ),                        // num items
    m_seq_nv(m_seq ? m_seq->shape[1]*m_seq->shape[2] : 0 ),        // num values in each item 
    m_seq_index(-1),
    m_cur(NP::Make<int>(m_seq_ni)),
    m_cur_values(m_cur->values<int>()),
    m_recycle(true)
{
}

inline std::string s_seq::desc() const 
{
    std::stringstream ss ; 
    ss << "s_seq::desc" 
       << std::endl 
       << " m_seqpath " << ( m_seqpath ? m_seqpath : "-" )
       << std::endl 
       << " m_seq " << ( m_seq ? m_seq->sstr() : "-" )
       << std::endl 
       ;
    std::string str = ss.str(); 
    return str ; 
}

inline double s_seq::flat()
{
    assert(m_seq_index > -1) ;  // must not call when disabled, use G4UniformRand to use standard engine
    int cursor = *(m_cur_values + m_seq_index) ;  // get the cursor value to use for this generation, starting from 0 

    if( cursor >= m_seq_nv )
    {   
        if(m_recycle == false)
        {   
            std::cout 
                << "s_seq::flat"
                << " FATAL : not enough precooked randoms and recycle not enabled "
                << " m_seq_index " << m_seq_index
                << " m_seq_nv " << m_seq_nv
                << " cursor " << cursor
                << std::endl
                ;
            assert(0);
        }
        else
        {   
            std::cout 
                << "s_seq::flat"
                << " WARNING : not enough precooked randoms are recycling randoms " 
                << " m_seq_index " << m_seq_index 
                << " m_seq_nv " << m_seq_nv 
                << " cursor " << cursor
                << std::endl 
                ;
            cursor = cursor % m_seq_nv ;
        }
    }


    int idx = m_seq_index*m_seq_nv + cursor ;

    float  f = m_seq_values[idx] ;
    double d = f ;               // promote random float to double 
    m_flat_prior = d ;

    *(m_cur_values + m_seq_index) += 1 ;          // increment the cursor in the array, for the next generation 

    return d ;
}

inline std::string s_seq::demo(int n) 
{
    std::stringstream ss ; 
    ss << "s_seq::demo m_seq_index " << m_seq_index << std::endl ;
    for(int i=0 ; i < n ; i++) ss 
         << std::setw(4) << i 
         << " : " 
         << std::fixed << std::setw(10) << std::setprecision(5) << flat() 
         << std::endl
         ; 
    std::string str = ss.str(); 
    return str ; 
}

inline int  s_seq::getSequenceIndex() const
{
    return m_seq_index ; 
}
inline void s_seq::setSequenceIndex(int index_)
{
    if( index_ < 0 )
    {
        m_seq_index = index_ ;
    }
    else
    {
        int idx = index_ ; // ASSUME NO MASKS 
        bool idx_in_range = int(idx) < m_seq_ni ;
        if(!idx_in_range)
            std::cout
                << "s_seq::setSequenceIndex"
                << "FATAL : OUT OF RANGE : "
                << " m_seq_ni " << m_seq_ni
                << " index_ " << index_
                << " idx " << idx << " (must be < m_seq_ni ) "
                << " desc "  << desc()
                ;
        assert( idx_in_range );
        m_seq_index = idx ;
    } 
}
inline bool s_seq::is_enabled() const
{
    return m_seq_index > -1 ; 
}


