#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdlib>

#include "Randomize.hh"
#include "G4Types.hh"
#include "U4Random.hh"
#include "NP.hh"

U4Random* U4Random::INSTANCE = nullptr ; 
U4Random* U4Random::Get(){ return INSTANCE ; }

const char* U4Random::NAME = "U4Random" ;  
const char* U4Random::OPTICKS_RANDOM_SEQPATH = "OPTICKS_RANDOM_SEQPATH" ; 

bool U4Random::Enabled()
{
    const char* seq = getenv(OPTICKS_RANDOM_SEQPATH) ; 
    return seq != nullptr ; 
}

/**
U4Random::U4Random
-------------------------------

When no seq path argument is provided the envvar OPTICKS_RANDOM_SEQPATH
to consulted to provide the path. 

The optional seqmask (a list of indices) allows working with 
sub-selections of the full set of streams of randoms. 
This allows reproducible running within photon selections
by arranging the same random stream to be consumed in 
full-sample and sub-sample running. 

Not that *seq* can either be the path to an .npy file
or the path to a directory containing .npy files which 
are concatenated using NP::Load/NP::Concatenate.

**/

U4Random::U4Random(const char* seq, const char* seqmask)
    :
    m_seqpath( seq ? seq : getenv(OPTICKS_RANDOM_SEQPATH) ), 
    m_seq(m_seqpath ? NP::Load(m_seqpath) : nullptr),
    m_seq_values(m_seq ? m_seq->cvalues<float>() : nullptr ),
    m_seq_ni(m_seq ? m_seq->shape[0] : 0 ),                        // num items
    m_seq_nv(m_seq ? m_seq->shape[1]*m_seq->shape[2] : 0 ),        // num values in each item 
    m_seq_index(-1),

    m_cur(NP::Make<int>(m_seq_ni)),
    m_cur_values(m_cur->values<int>()),
    m_recycle(true),
    m_default(CLHEP::HepRandom::getTheEngine()),

    m_seqmask(seqmask ? NP::Load(seqmask) : nullptr),
    m_seqmask_ni( m_seqmask ? m_seqmask->shape[0] : 0 ),
    m_seqmask_values(m_seqmask ? m_seqmask->cvalues<size_t>() : nullptr),
    m_flat_debug(false),
    m_flat_prior(0.)
{
    INSTANCE = this ; 
    bool has_seq = m_seq != nullptr ; 
    if(has_seq == false)
        std::cerr 
            << "U4Random::U4Random"
            << " FATAL : FAILED TO LOAD SINGLE .npy OR DIRECTORY OF .npy FROM " 
            << " m_seqpath " << ( m_seqpath ? m_seqpath : "-" )  
            << " ekey OPTICKS_RANDOM_SEQPATH "
            << " generate the precooked randoms with cd ~/opticks/qudarap/tests ; ./rng_sequence.sh run "
            << std::endl 
            ;
 
    std::cout << detail() ; 
    assert(has_seq); 
}

std::string U4Random::detail() const 
{
    std::stringstream ss ; 
    ss << "U4Random::detail"
       << " m_seq " << ( m_seq ? m_seq->desc() : "-" ) << std::endl 
       << " m_seqmask " << ( m_seqmask ? m_seqmask->desc() : "-" ) << std::endl
       << " desc " << desc() << std::endl 
       << " m_cur " << ( m_cur ? m_cur->desc() : "-" ) << std::endl 
       ;
    std::string s = ss.str(); 
    return s ; 
}


/**
U4Random::getNumIndices
-----------------------------

With seqmask running returned the number of seqmask indices otherwise returns the total number of indices. 
This corresponds to the total number of available streams of randoms. 

**/

size_t U4Random::getNumIndices() const
{
   return m_seq && m_seqmask ? m_seqmask_ni : ( m_seq ? m_seq_ni : 0 ) ; 
}

/**
U4Random::SetSeed
-----------------------

static control of the seed, NB calling this while enabled will assert 
as there is no role for a seed with pre-cooked randoms

**/

void U4Random::SetSeed(long seed)  // static
{
    CLHEP::HepRandomEngine* engine = CLHEP::HepRandom::getTheEngine(); 
    int dummy = 0 ; 
    engine->setSeed(seed, dummy); 
}

/**
U4Random::getMaskedIndex
------------------------------

When no seqmask is active this just returns the argument.
When a seqmask selection is active indices from the mask are returned.

**/

size_t U4Random::getMaskedIndex(int index_)
{
    if( m_seqmask == nullptr  ) return index_ ; 
    assert( index_ < m_seqmask_ni ); 
    size_t idx = m_seqmask_values[index_] ;  
    return idx ; 
}


int U4Random::getSequenceIndex() const 
{
    return m_seq_index ; 
}

/**
U4Random::setSequenceIndex
--------------------------------

Switches random stream when index is not negative.
This is used for example used to switch between the separate streams 
used for each photon.

A negative index disables the control of the Geant4 random engine.  

**/

void U4Random::setSequenceIndex(int index_)
{
    if( index_ < 0 )
    {
        m_seq_index = index_ ; 
        disable() ;
    }
    else
    {
        size_t idx = getMaskedIndex(index_); 
        bool idx_in_range = int(idx) < m_seq_ni ; 

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


std::string U4Random::desc() const
{
    std::stringstream ss ; 
    ss << " m_seq_ni " << m_seq_ni << " m_seq_nv " << m_seq_nv ; 
    return ss.str();
}


U4Random::~U4Random()
{
}

/**
U4Random::enable
----------------------

Invokes CLHEP::HepRandom::setTheEngine to *this* U4Random instance 
which means that all subsequent calls to G4UniformRand will provide pre-cooked 
randoms from the stream controlled by *U4Random::setSequenceIndex*

**/

void U4Random::enable()
{
    CLHEP::HepRandom::setTheEngine(this); 
}

/**
U4Random::disable
-----------------------

Returns Geant4 to using to the default engine. 

**/

void U4Random::disable()
{
    CLHEP::HepRandom::setTheEngine(m_default); 
}


/**
U4Random::dump
----------------------

Invokes G4UniformRand *n* times dumping the values. 

**/

void U4Random::dump(unsigned n)
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


/**
U4Random::flat
--------------------

This is the engine method that gets invoked by G4UniformRand calls 
and which returns pre-cooked randoms. 
The *m_cur_values* cursor is updated to maintain the place in the sequence. 

**/

double U4Random::flat()
{
    assert(m_seq_index > -1) ;  // must not call when disabled, use G4UniformRand to use standard engine

    int cursor = *(m_cur_values + m_seq_index) ;  // get the cursor value to use for this generation, starting from 0 

    if( cursor >= m_seq_nv )
    {
        if(m_recycle == false)
        {
            std::cout 
                << "U4Random::flat"
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
                << "U4Random::flat"
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

    if( m_flat_debug )
    {
        std::cout 
            << "U4Random::flat "
            << " m_seq_index " << std::setw(4) << m_seq_index
            << " m_seq_nv " << std::setw(4) << m_seq_nv
            << " cursor " << std::setw(4) << cursor 
            << " idx " << std::setw(4) << idx 
            << " d " <<  std::setw(10 ) << std::fixed << std::setprecision(5) << d 
            << std::endl 
            ;
    }

    m_flat_prior = d ; 
    return d ; 
}


double U4Random::getFlatPrior() const 
{
    return m_flat_prior ; 
}


/**
U4Random::flatArray
--------------------------

This method and several others are required as U4Random ISA CLHEP::HepRandomEngine

**/

void U4Random::flatArray(const int size, double* vect)
{
     assert(0); 
}
void U4Random::setSeed(long seed, int)
{
    assert(0); 
}
void U4Random::setSeeds(const long * seeds, int)
{
    assert(0); 
}
void U4Random::saveStatus( const char filename[]) const 
{
    assert(0); 
}
void U4Random::restoreStatus( const char filename[]) 
{
    assert(0); 
}
void U4Random::showStatus() const 
{
    assert(0); 
}
std::string U4Random::name() const 
{
    return NAME ; 
}

