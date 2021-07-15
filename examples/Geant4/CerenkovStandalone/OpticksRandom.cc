#include <iostream>
#include <sstream>
#include "Randomize.hh"
#include "G4Types.hh"
#include "OpticksRandom.hh"
#include "NP.hh"

OpticksRandom* OpticksRandom::INSTANCE = nullptr ; 
OpticksRandom* OpticksRandom::Get(){ return INSTANCE ; }

const char* OpticksRandom::NAME = "OpticksRandom" ;  

OpticksRandom::OpticksRandom(const char* path)
    :
    m_seq(NP::Load(path)),
    m_seq_values(m_seq ? m_seq->values<double>() : nullptr ),
    m_seq_ni(m_seq ? m_seq->shape[0] : 0 ),
    m_seq_nv(m_seq ? m_seq->shape[1]*m_seq->shape[2] : 0 ),
    m_seq_index(-1),
    m_cur(NP::Make<int>(m_seq_ni)),
    m_cur_values(m_cur->values<int>()),
    m_recycle(true),
    m_default(CLHEP::HepRandom::getTheEngine())
{
    INSTANCE = this ; 
    if( m_seq )
    {
        std::cout << " loaded " << path << std::endl ; 
        std::cout << " m_seq " << ( m_seq ? m_seq->desc() : "-" ) << std::endl ; 
        std::cout << " desc " << desc() << std::endl ; 
        std::cout << " m_cur " << ( m_cur ? m_cur->desc() : "-" ) << std::endl ; 
    }
}


void OpticksRandom::setSequenceIndex(int seq_index)
{
    bool have_seq = seq_index < m_seq_ni ; 
    if(!have_seq) 
        std::cout 
            << "FATAL : OUT OF RANGE : " 
            << " m_seq_ni " << m_seq_ni 
            << " seq_index " << seq_index << " (must be < m_seq_ni ) "  
            << " desc "  << desc()
            ; 
    assert( have_seq );
  
    m_seq_index = seq_index ; 

    if( m_seq_index < 0)  
    {   
        disable(); 
    }   
    else 
    {   
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

    double u = m_seq_values[idx] ;

    *(m_cur_values + m_seq_index) += 1 ;          // increment the cursor in the array, for the next generation 

    return u ; 
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




