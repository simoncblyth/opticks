#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdlib>

#include "Randomize.hh"
#include "G4Types.hh"
#include "OpticksRandom.hh"
#include "OpticksUtil.hh"
#include "NP.hh"

OpticksRandom* OpticksRandom::INSTANCE = nullptr ; 
OpticksRandom* OpticksRandom::Get(){ return INSTANCE ; }

const char* OpticksRandom::NAME = "OpticksRandom" ;  
const char* OpticksRandom::OPTICKS_RANDOM_SEQPATH = "OPTICKS_RANDOM_SEQPATH" ; 

bool OpticksRandom::Enabled()
{
    const char* seq = getenv(OPTICKS_RANDOM_SEQPATH) ; 
    return seq != nullptr ; 
}

/**
OpticksRandom::OpticksRandom
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
are concatenated using OpticksUtil::LoadConcat/NP::Concatenate.

**/

OpticksRandom::OpticksRandom(const char* seq, const char* seqmask)
    :
    m_seqpath( seq ? seq : getenv(OPTICKS_RANDOM_SEQPATH) ), 
    m_seq(m_seqpath ? OpticksUtil::LoadConcat(m_seqpath) : nullptr),
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
    std::cout << detail() ; 
}

std::string OpticksRandom::detail() const 
{
    std::stringstream ss ; 
    ss << "OpticksRandom::detail"
       << " m_seq " << ( m_seq ? m_seq->desc() : "-" ) << std::endl 
       << " m_seqmask " << ( m_seqmask ? m_seqmask->desc() : "-" ) << std::endl
       << " desc " << desc() << std::endl 
       << " m_cur " << ( m_cur ? m_cur->desc() : "-" ) << std::endl 
       ;
    std::string s = ss.str(); 
    return s ; 
}


/**
OpticksRandom::getNumIndices
-----------------------------

With seqmask running returned the number of seqmask indices otherwise returns the total number of indices. 
This corresponds to the total number of available streams of randoms. 

**/

size_t OpticksRandom::getNumIndices() const
{
   return m_seq && m_seqmask ? m_seqmask_ni : ( m_seq ? m_seq_ni : 0 ) ; 
}

/**
OpticksRandom::SetSeed
-----------------------

static control of the seed, NB calling this while enabled will assert 
as there is no role for a seed with pre-cooked randoms

**/

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

/**
OpticksRandom::setSequenceIndex
--------------------------------

Switches random stream when index is not negative.
This is used for example used to switch between the separate streams 
used for each photon.

A negative index disables the control of the Geant4 random engine.  

**/

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

/**
OpticksRandom::enable
----------------------

Invokes CLHEP::HepRandom::setTheEngine to *this* OpticksRandom instance 
which means that all subsequent calls to G4UniformRand will provide pre-cooked 
randoms from the stream controlled by *OpticksRandom::setSequenceIndex*

**/

void OpticksRandom::enable()
{
    CLHEP::HepRandom::setTheEngine(this); 
}

/**
OpticksRandom::disable
-----------------------

Returns Geant4 to using to the default engine. 

**/

void OpticksRandom::disable()
{
    CLHEP::HepRandom::setTheEngine(m_default); 
}


/**
OpticksRandom::dump
----------------------

Invokes G4UniformRand *n* times dumping the values. 

**/

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


/**
OpticksRandom::flat
--------------------

This is the engine method that gets invoked by G4UniformRand calls 
and which returns pre-cooked randoms. 
The *m_cur_values* cursor is updated to maintain the place in the sequence. 

**/

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

    if( m_flat_debug )
    {
        std::cout 
            << "OpticksRandom::flat "
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


double OpticksRandom::getFlatPrior() const 
{
    return m_flat_prior ; 
}


/**
OpticksRandom::flatArray
--------------------------

This method and several others are required as OpticksRandom ISA CLHEP::HepRandomEngine

**/

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

