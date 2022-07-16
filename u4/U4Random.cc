#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <csignal>

#include "Randomize.hh"
#include "G4Types.hh"
#include "U4Random.hh"
#include "U4Stack.h"
#include "U4StackAuto.h"

#include "NP.hh"
#include "SPath.hh"
#include "SEvt.hh"
#include "SSys.hh"

#include "SBacktrace.h"
#include "SDBG.h"

#include "PLOG.hh"

const plog::Severity U4Random::LEVEL = PLOG::EnvLevel("U4Random", "DEBUG") ; 

U4Random* U4Random::INSTANCE = nullptr ; 
U4Random* U4Random::Get(){ return INSTANCE ; }

const char* U4Random::NAME = "U4Random" ;  


void U4Random::SetSequenceIndex(int index)
{
    if(U4Random::Get() == nullptr) return ; 

    U4Random* rnd = U4Random::Get();  
    if(rnd->isReady() == false) 
    {
        LOG(LEVEL) << " index " << index << " NOT READY " ; 
        return ; 
    }
    rnd->setSequenceIndex(index);  
}

int U4Random::GetSequenceIndex()
{
    if(U4Random::Get() == nullptr) return -1 ; 
    U4Random* rnd = U4Random::Get();  
    return rnd->isReady() ? rnd->getSequenceIndex() : -1 ; 
}




const char* U4Random::SeqPath(){ return SSys::getenvvar(OPTICKS_RANDOM_SEQPATH, DEFAULT_SEQPATH ) ; } // static  



/**
U4Random::U4Random
-------------------------------

When no seq path argument is provided the envvar OPTICKS_RANDOM_SEQPATH
is consulted to provide the path. 

The optional seqmask (a list of size_t or "unsigned long long" indices) 
allows working with sub-selections of the full set of streams of randoms. 
This allows reproducible running within photon selections
by arranging the same random stream to be consumed in 
full-sample and sub-sample running. 

Not that *seq* can either be the path to an .npy file
or the path to a directory containing .npy files which 
are concatenated using NP::Load/NP::Concatenate.

TODO: when next need to use seqmask, change the interface to enabling it 
to be envvar OR SEventConfig based and eliminate the U4Random arguments 

**/




U4Random::U4Random(const char* seq, const char* seqmask)
    :
    m_seqpath(SPath::Resolve( seq ? seq : SeqPath(), NOOP)), 
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
    //m_flat_debug(SSys::getenvbool("U4Random_flat_debug")),
    m_flat_prior(0.),
    m_ready(false),
    m_select(SSys::getenvintvec("U4Random_select")),
    m_select_action(SDBG::Action(SSys::getenvvar("U4Random_select_action", "backtrace")))   // "backtrace" "caller" "interrupt" "summary"
{
    init(); 
}

/**
U4Random::isSelect
---------------------

Returns true when the (photon index, random index "flat cursor") tuple
provided in the arguments matches one of the pairs provided in the 
comma delimited U4Random_select envvar.

An U4Random_select envvar of (-1,-1) is special cased to always match, 
and hence may generate very large amounts of output dumping. 
A single -1 corresponds to a wildcard:

* (0,-1) will match all cursors for photon idx 0 
* (-1,0) will match all cursor 0 for all photon idx 


This allows selection of one or more U4Random::flat calls.
For example std::raise(SIGINT) could be called when a random draw of 
interest is done in order to examine the call stack for that random 
consumption before resuming processing. 

**/

bool U4Random::isSelect(int photon_idx, int flat_cursor) const 
{
   if(m_select == nullptr) return false ; 
   assert( m_select->size() % 2 == 0 );  
   for(unsigned p=0 ; p < m_select->size()/2 ; p++)
   {
       int _pidx   = (*m_select)[p*2+0] ; 
       int _cursor = (*m_select)[p*2+1] ; 
       bool match = (( _pidx == photon_idx || _pidx == -1)  && (_cursor == flat_cursor || _cursor == -1)) ;
       if(match) return true ;  
   }
   return false ; 
}

std::string U4Random::descSelect(int photon_idx, int flat_cursor ) const
{
    std::stringstream ss ; 
    ss << "U4Random_select " << SSys::getenvvar("U4Random_select", "-") 
       << " m_select->size " << ( m_select ? m_select->size() : 0 )   
       ;

    if(m_select) 
    {
        for(unsigned p=0 ; p < m_select->size()/2 ; p++)
        {
           int _pidx   = (*m_select)[p*2+0] ; 
           int _cursor = (*m_select)[p*2+1] ; 
           bool match = ( ( _pidx == photon_idx || _pidx == -1)  && (_cursor == flat_cursor || _cursor == -1) ) ; 
           ss << " (" << _pidx << "," << _cursor << ") " << ( match ? "YES" : "NO" )  << " " ; 
        }
    }
    std::string s = ss.str(); 
    return s ; 
}



void U4Random::init()
{
    INSTANCE = this ; 
    m_ready = m_seq != nullptr ; 
    if(m_ready == false) 
        LOG(error)
            << desc()
            << std::endl 
            << NOTES
            << std::endl 
            ;
}

bool U4Random::isReady() const { return m_ready ; }

std::string U4Random::desc() const
{
    std::stringstream ss ; 
    ss 
         << "U4Random::desc"
         << " U4Random::isReady() " << ( m_ready ? "YES" : "NO" )
         << " m_seqpath " << ( m_seqpath ? m_seqpath : "-" )
         << " m_seq_ni " << m_seq_ni 
         << " m_seq_nv " << m_seq_nv 
         ; 
    return ss.str();
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

TODO: THIS IS RATHER OUT OF PLACE AS NOT MUCH RELATED TO ALIGNED RUNNING, SO RELOCATE TO SEPARATE STRUCT "U4HepRandomEngine::SetSeed" ?

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

Masked running allows to reproduce running on subsets of photons from 
a larger sample. 

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
    LOG(LEVEL) << " index " << index_ ; 

    if( index_ < 0 )
    {
#ifdef DEBUG_TAG
        check_cursor_vs_tagslot() ; 
#endif
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
            /*
            std::cout 
                << "U4Random::flat"
                << " WARNING : not enough precooked randoms are recycling randoms " 
                << " m_seq_index " << m_seq_index 
                << " m_seq_nv " << m_seq_nv 
                << " cursor " << cursor
                << std::endl 
                ;
            */
            cursor = cursor % m_seq_nv ; 
        }
    }



    int idx = m_seq_index*m_seq_nv + cursor ;

    float  f = m_seq_values[idx] ;
    double d = f ;     // promote random float to double 
    m_flat_prior = d ; 

    *(m_cur_values + m_seq_index) += 1 ;          // increment the cursor in the array, for the next generation 

    if(m_seq_index == SEvt::PIDX)
    { 
        std::cout << "-------U4Random::flat" << std::endl << std::endl ; 
        LOG(info)
            << " SEvt::PIDX " << SEvt::PIDX
            << " m_seq_index " << std::setw(4) << m_seq_index
            << " m_seq_nv " << std::setw(4) << m_seq_nv
            << " cursor " << std::setw(4) << cursor 
            << " idx " << std::setw(4) << idx 
            << " d " <<  std::setw(10 ) << std::fixed << std::setprecision(5) << d 
            ;

        char* summary = SBacktrace::Summary(); 

        LOG(info) << std::endl << summary ; 

    }


    bool auto_tag = false ;   // unfortunately stack summaries lack vital lines on Linux 

    if( auto_tag )
    {
        char* summary = SBacktrace::Summary(); 
        unsigned stack = U4StackAuto::Classify(summary); 
        bool is_classified = U4StackAuto::IsClassified(stack) ; 

        //LOG(info) << " stack " << std::setw(2) << stack << " " << U4Stack::Name(stack) ; 

        if(is_classified == false) LOG(error) << std::endl << summary ; 

        bool select = isSelect(m_seq_index, cursor ) || is_classified == false ; 
        if( select ) 
        {
            LOG(info) << descSelect(m_seq_index, cursor) ; 
            switch(m_select_action)
            {
                case SDBG::INTERRUPT: std::raise(SIGINT)        ; break ; 
                case SDBG::BACKTRACE: SBacktrace::Dump()        ; break ; 
                case SDBG::CALLER:    SBacktrace::DumpCaller()  ; break ; 
                case SDBG::SUMMARY:   SBacktrace::DumpSummary() ; break ; 
            }
        }

        SEvt::AddTag(stack, f );  
    }


    return d ; 
}


#ifdef DEBUG_TAG
/**
U4Random::check_cursor_vs_tagslot
----------------------------------

This is called by setSequenceIndex with index -1 signalling the end 
of the index. A comparison between the below counts is made:

* number of randoms provided by U4Random::flat for the last m_seq_index as indicated by the cursor 
* random consumption tags added with SEvt::AddTag

**/

void U4Random::check_cursor_vs_tagslot() 
{
    assert(m_seq_index > -1) ;  // must not call when disabled, use G4UniformRand to use standard engine
    int cursor = *(m_cur_values + m_seq_index) ;  // get the cursor value to use for this generation, starting from 0 


    int slot = SEvt::GetTagSlot(); 
    bool cursor_slot_match = cursor == slot ;  

    //LOG(info) << " m_seq_index " << m_seq_index << " cursor " << cursor << " slot " << slot << " cursor_slot_match " << cursor_slot_match ; 

    if(!cursor_slot_match) 
    {
        m_problem_idx.push_back(m_seq_index); 
        LOG(error) 
            << " m_seq_index " << m_seq_index 
            << " cursor " << cursor 
            << " slot " << slot 
            << " cursor_slot_match " << cursor_slot_match  
            << std::endl 
            << " PROBABLY SOME RANDOM CONSUMPTION LACKS SEvt::AddTag CALLS "
            ; 
    } 
    assert( cursor_slot_match ); 
}
#endif


void U4Random::saveProblemIdx(const char* fold) const 
{
    std::cout << "U4Random::saveProblemIdx m_problem_idx.size " << m_problem_idx.size() << " (" ; 
    for(unsigned i=0 ; i <  m_problem_idx.size() ; i++ ) std::cout << m_problem_idx[i] << " " ; 
    std::cout << ")" << std::endl ; 

    NP::Write<int>( fold, "problem_idx.npy", m_problem_idx ); 
}




/**
U4Random::getFlatTag
----------------------------

By examination of the SBacktrace and Geant4 process state etc.. 
need to determine what is the random consumer and assign this consumption 
of a random number with a standardized tag enumeration from stag.h   
to facilitate random alignment with the GPU simulation. 

::

   u4t ; U4Random_select=-1,0 U4Random_select_action=backtrace ./U4RecorderTest.sh run
       dump full backtraces for first flat consumption of all photons 

   u4t ; U4Random_select=-1,0 U4Random_select_action=summary ./U4RecorderTest.sh run
       dump summary backtraces for first flat consumption of all photons 


unsigned U4Random::getFlatTag() 
{
    char* summary = SBacktrace::Summary(); 
    unsigned stk = U4Stack::Classify(summary); 
    LOG(info) << " stk " << std::setw(2) << stk << " U4Stack::Name(stk) " << U4Stack::Name(stk) ;  
    if(!U4Stack::IsClassified(stk)) LOG(error) << std::endl << summary ; 
    return stk  ; // stk needs translation into the stag.h enumeration 
}
**/



int U4Random::getFlatCursor() const 
{
    if(m_seq_index < 0) return -1 ; 
    int cursor = *(m_cur_values + m_seq_index) ;  // get the cursor value to use for this generation, starting from 0 
    return cursor ; 
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

