#include <iostream>
#include <sstream>
#include "Randomize.hh"
#include "G4Types.hh"
#include "OpticksRandom.hh"
#include "NP.hh"

OpticksRandom* OpticksRandom::INSTANCE = nullptr ; 
OpticksRandom* OpticksRandom::Get(){ return INSTANCE ; }

const char* OpticksRandom::NAME = "OpticksRandom" ;  

OpticksRandom::OpticksRandom(const NP* seq, const NP* seqmask)
    :
    m_seq(seq),
    m_seq_values(m_seq ? m_seq->cvalues<float>() : nullptr ),
    m_seq_ni(m_seq ? m_seq->shape[0] : 0 ),
    m_seq_nv(m_seq ? m_seq->shape[1]*m_seq->shape[2] : 0 ),
    m_seq_index(-1),
    m_seqmask(seqmask),
    m_seqmask_ni( m_seqmask ? m_seqmask->shape[0] : 0 ),
    m_seqmask_values(m_seqmask ? m_seqmask->cvalues<size_t>() : nullptr), 
    m_cur(NP::Make<int>(m_seq_ni)),
    m_cur_values(m_cur->values<int>()),
    m_recycle(true),
    m_default(CLHEP::HepRandom::getTheEngine())
{
    INSTANCE = this ; 
    if( m_seq )
    {
        std::cout << " m_seq " << ( m_seq ? m_seq->desc() : "-" ) << std::endl ; 
        std::cout << " m_seqmask " << ( m_seqmask ? m_seqmask->desc() : "-" ) << std::endl ; 
        std::cout << " desc " << desc() << std::endl ; 
        std::cout << " m_cur " << ( m_cur ? m_cur->desc() : "-" ) << std::endl ; 
    }
}


size_t OpticksRandom::getNumIndices() const
{
   return m_seq && m_seqmask ? m_seqmask_ni : ( m_seq ? m_seq_ni : 0 ) ; 
}

void OpticksRandom::SetSeed(long seed)  // static
{
    CLHEP::HepRandomEngine* engine = CLHEP::HepRandom::getTheEngine(); 
    int dummy = 0 ; 
    engine->setSeed(seed, dummy); 
}


/**
OpticksRandom::getMaskedIndex
------------------------------

When no seqmask is active this just returns the argument.
When a seqmask selection is active indices from the mask are returned.

**/

size_t OpticksRandom::getMaskedIndex(int index_)
{
    if( m_seqmask == nullptr  ) return index_ ; 
    assert( index_ < m_seqmask_ni ); 
    size_t idx = m_seqmask_values[index_] ;  
    return idx ; 
}

int OpticksRandom::getSequenceIndex() const 
{
    return m_seq_index ; 
}
void OpticksRandom::setSequenceIndex(int index_)
{
    if( index_ < 0 )
    {
        m_seq_index = index_ ; 
        disable() ;
    }
    else
    {
        size_t idx = getMaskedIndex(index_); 
        bool idx_in_range = idx < m_seq_ni ; 

        if(!idx_in_range) 
            std::cout 
                << "FATAL : OUT OF RANGE : " 
                << " m_seq_ni " << m_seq_ni 
                << " index_ " << index_ 
                << " idx " << idx << " (must be < m_seq_ni ) "  
                << " desc "  << desc()
                ; 
        assert( idx_in_range );
        m_seq_index = idx ; 
        enable();
    }   
}


std::string OpticksRandom::desc() const
{
    std::stringstream ss ; 
    ss << " m_seq_ni " << m_seq_ni << " m_seq_nv " << m_seq_nv ; 
    return ss.str();
}


OpticksRandom::~OpticksRandom()
{
}

void OpticksRandom::enable()
{
    CLHEP::HepRandom::setTheEngine(this); 
}
void OpticksRandom::disable()
{
    CLHEP::HepRandom::setTheEngine(m_default); 
}

void OpticksRandom::dump(unsigned n)
{
    for(unsigned i=0 ; i < n ; i++)
    {
        G4double u = G4UniformRand() ;   
        std::cout 
            << " i " << std::setw(5) << i 
            << " u " << std::fixed << std::setw(10) << std::setprecision(5) << u 
            << std::endl 
            ;            
    }
}

double OpticksRandom::flat()
{
    assert(m_seq_index > -1) ;  // must not call when disabled, use G4UniformRand to use standard engine

    int cursor = *(m_cur_values + m_seq_index) ;  // get the cursor value to use for this generation, starting from 0 

    if( cursor >= m_seq_nv )
    {
        if(m_recycle == false)
        {
            std::cout 
                << "OpticksRandom::flat"
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
                << "OpticksRandom::flat"
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
    double d = f ;     // promote random float to double 

    *(m_cur_values + m_seq_index) += 1 ;          // increment the cursor in the array, for the next generation 

    return d ; 
}



void OpticksRandom::flatArray(const int size, double* vect)
{
     assert(0); 
}
void OpticksRandom::setSeed(long seed, int)
{
    assert(0); 
}
void OpticksRandom::setSeeds(const long * seeds, int)
{
    assert(0); 
}
void OpticksRandom::saveStatus( const char filename[]) const 
{
    assert(0); 
}
void OpticksRandom::restoreStatus( const char filename[]) 
{
    assert(0); 
}
void OpticksRandom::showStatus() const 
{
    assert(0); 
}
std::string OpticksRandom::name() const 
{
    return NAME ; 
}




