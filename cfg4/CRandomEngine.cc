/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */


#include <iomanip>
#include <csignal>

#include "PLOG.hh"

#include "Randomize.hh"
#include "G4String.hh"
#include "G4VProcess.hh"


#include "SVec.hh"
#include "SSys.hh"

#include "BStr.hh"
#include "BLog.hh"
#include "BFile.hh"
#include "BLocSeq.hh"

#include "Opticks.hh"
#include "OpticksRun.hh"
#include "OpticksEvent.hh"
#include "OpticksFlags.hh"
#include "OpticksSwitches.h"

#include "NPY.hpp"

#include "CManager.hh"
#include "CCtx.hh"
#include "CProcess.hh"
#include "CProcessManager.hh"
#include "CStepStatus.hh"
#include "CStepping.hh"
#include "CRandomEngine.hh"

#ifdef DYNAMIC_CURAND
#include "TCURAND.hh"
#endif

const plog::Severity CRandomEngine::LEVEL = PLOG::EnvLevel("CRandomEngine", "DEBUG") ; 



std::string CRandomEngine::name() const 
{
    return "CRandomEngine";
}


int CRandomEngine::preinit()
{
    OK_PROFILE("_CRandomEngine::CRandomEngine"); 
    return 0 ; 
}

CRandomEngine::CRandomEngine(CManager* manager)
    :
    m_manager(manager),
    m_ctx(manager->getCtx()),
    m_ok(m_ctx.getOpticks()),
    m_dbgflat(m_ok->isDbgFlat()),    // --dbgflat
    m_curflatsigint(m_ok->getCurFlatSigInt()), // --curflatsigint N
    m_preinit(preinit()),
    m_dbgkludgeflatzero(m_ok->isDbgKludgeFlatZero()),    // --dbgkludgeflatzero
    m_run(nullptr),
    m_okevt(NULL),
    m_okevt_seqhis(0),
    m_okevt_pt(NULL),
    m_pindexlogpath(NULL),
    m_pindexlog(NULL),
    m_dbgflatlog(NULL),
    m_g4evt(NULL),
    m_mask(m_ok->getMask()),
    m_masked(m_mask.size() > 0),
    m_alignlevel(m_ok->getAlignLevel()),
    m_seed(9876),
    m_internal(false),
    m_skipdupe(true),
    m_locseq(m_alignlevel > 1 ? new BLocSeq<unsigned long long>(m_skipdupe) : NULL ),
    m_tranche_size(100000),
    m_tranche_id(-1),
    m_tranche_ibase(-1),
    m_tranche_index(-1),
#ifdef DYNAMIC_CURAND
    m_precurand(precurand()),
    m_tcurand(new TCURAND<double>(m_tranche_size,16,16)),
    m_postcurand(postcurand()),
    m_seq(m_tcurand->getArray()),
#else
    m_path("$TMP/TRngBufTest_0.npy"),
    m_seq(NPY<double>::load(m_path)),
#endif
    m_seq_ni(m_seq ? m_seq->getShape(0) : 0 ),
    m_seq_nv(m_seq ? m_seq->getNumValues(1) : 0 ),  // itemvalues
    m_current_record_flat_count(0),
    m_current_step_flat_count(0),
    m_jump(0),
    m_jump_count(0),
    m_flat(-1.0),
    m_cursor(-1),
    m_cursor_old(-1),
    m_setupTranche_acc(m_ok->accumulateAdd("CRandomEngine::setupTranche"))
{
    init();
}

void CRandomEngine::init()
{
    OK_PROFILE("CRandomEngine::CRandomEngine"); 
    LOG(LEVEL) << "["; 
    initCurand();
    CLHEP::HepRandom::setTheEngine( this );  
    LOG(LEVEL) << "]"; 
}


int CRandomEngine::precurand()
{
    OK_PROFILE("_TCURAND::TCURAND"); 
    return 0 ; 
}
int CRandomEngine::postcurand()
{
    OK_PROFILE("TCURAND::TCURAND"); 
    return 0 ; 
}




bool CRandomEngine::hasSequence() const 
{
    return m_seq && m_seq_ni > 0 && m_seq_nv > 0 ; 
}

#ifdef DYNAMIC_CURAND
#else
const char* CRandomEngine::getPath() const 
{
    return m_path ; 
}
#endif


void CRandomEngine::dumpDouble(const char* msg, double* v, unsigned width ) const 
{
    LOG(info) << msg ; 
    assert( m_seq_nv > 15 );
    for(int i=0 ; i < 16 ; i++)  
    {
        std::cout << std::fixed << std::setw(10) << std::setprecision(10) << v[i] << " " ; 
        if( i % width == (width - 1) ) std::cout << std::endl ; 
    }
}


void CRandomEngine::initCurand()
{
    LOG(LEVEL) 
#ifdef DYNAMIC_CURAND
        << " DYNAMIC_CURAND " 
#else
        << " STATIC_CURAND path " << m_path  
#endif
        << ( m_seq ? m_seq->getShapeString() : "-" ) 
        << " curand_ni " << m_seq_ni
        << " curand_nv " << m_seq_nv
        ; 

#ifdef DYNAMIC_CURAND
#else
    checkTranche();  
#endif
}

void CRandomEngine::checkTranche()
{
    if(!m_seq) return ; 
    assert( m_seq->hasShape(m_tranche_size,16,16)) ; 
}

void CRandomEngine::saveTranche()
{
    std::string path  = m_tcurand->getRuncachePath(m_ok->getRuncacheDir());
    LOG(LEVEL) << " save to " << path ; 
    m_tcurand->save(path.c_str()); 
}



void CRandomEngine::dumpTranche()
{
    if(!m_seq) return ; 

    assert( m_seq->hasShape(m_tranche_size,16,16)) ; 
        
    unsigned w = 4 ; 
    if( m_seq_ni > 0 )
         dumpDouble( "v0" , m_seq->getValues(0), w ) ; 

    if( m_seq_ni > 1 )
         dumpDouble( "v1" , m_seq->getValues(1), w ) ; 

    if( m_seq_ni > m_tranche_size - 1  )
        dumpDouble( "v-1" , m_seq->getValues(m_tranche_size - 1), w ) ; 
}


#ifdef DYNAMIC_CURAND
/**
CRandomEngine::setupTranche
-------------------------------

Invoked from setupCurandSequence when a photon record_id
which is not within the current tranche.

HMM potential for very inefficient if G4 photon record_id 
jumps around between tranches

Actually aligned running needs to use input photons
anyhow, so could restrict the number of input photons
to fit within one tranche.

HMM i dont like the hidden nature of this GPU launch 

**/

void CRandomEngine::setupTranche(int tranche_id)
{
    m_ok->accumulateStart(m_setupTranche_acc) ; 
    OK_PROFILE("_CRandomEngine::setupTranche"); 

    m_tranche_id = tranche_id ; 
    m_tranche_ibase = m_tranche_id*m_tranche_size ; 

    LOG(LEVEL) 
        << " DYNAMIC_CURAND "
        << " m_tranche_id " << m_tranche_id 
        << " m_tranche_size " << m_tranche_size
        << " m_tranche_ibase " << m_tranche_ibase
        ;

    m_tcurand->setIBase(m_tranche_ibase);   // <-- does GPU launch to init curand and generate the randoms
    checkTranche();

    OK_PROFILE("CRandomEngine::setupTranche"); 
    m_ok->accumulateStop(m_setupTranche_acc) ; 
}
#endif


/**
CRandomEngine::setupCurandSequence
-----------------------------------

Invoked from preTrack to prepare the random sequence for 
a single photon.

STATIC_CURAND 
    limited to the 16*16*100k randoms precooked for 100k photons by TRngBufTest 

DYNAMIC_CURAND 
    removes the limitation using TCURAND to dynamically generate the 16*16*100k 
    randoms for 100k photons in tranches 


Note that CRandomEngine::_flat increments the cursor 
and asserts that it is within the sequence size.  Suspect 
vague memory of re-cycling sequence was from before 
increased to 16*16 = 256 randoms for each photon.

**/

void CRandomEngine::setupCurandSequence(int record_id)
{
#ifdef DYNAMIC_CURAND
    int tranche_id = record_id/m_tranche_size ; 
    if( tranche_id != m_tranche_id ) // <-- TODO: check do not get flip-flop between tranches 
    {
        setupTranche(tranche_id); 
    }
    m_tranche_index = record_id - m_tranche_ibase ; 
    // dynamically generates the randoms in this tranche, so will always be in range 
#else
    assert( m_seq_ni > 0 && " no precooked RNG loaded see : TRngBufTest " );
    bool in_range = record_id > -1 && record_id < m_seq_ni ; 
    if(!in_range) LOG(fatal) << " OUT OF RANGE " << " record_id " << record_id << " m_seq_ni " << m_seq_ni ; 

    assert( in_range ); 
    assert( m_seq_nv > 0 ) ;

    m_tranche_index = record_id ; 
#endif

    LOG(LEVEL) 
        << " record_id " << record_id
        << " m_tranche_id " << m_tranche_id
        << " m_tranche_size " << m_tranche_size
        << " m_tranche_index " << m_tranche_index
        << " m_seq_ni " << m_seq_ni
        << " m_seq_nv " << m_seq_nv
        ; 

    double* seq = m_seq->getValues(m_tranche_index) ; 

    setRandomSequence( seq, m_seq_nv ) ; 

    m_current_record_flat_count = 0 ; 
    m_current_step_flat_count = 0 ; 
}


std::string CRandomEngine::desc() const 
{
    std::stringstream rs1_ ; 
    rs1_ <<  m_ctx._record_id  << "." << ( m_ctx._step_id + 1 ) ; 
    std::string rs1 = rs1_.str() ; 

    std::stringstream ss ; 
    ss 
       << "CRandomEngine"
       << " rec.stp1 " << std::setw(5) << rs1
       << " crf " << std::setw(5) << m_current_record_flat_count 
       << " csf " << std::setw(5) << m_current_step_flat_count 
       << " loc " << std::setw(50) << m_location 
       ;

    return ss.str();
}


/**
CRandomEngine::CurrentGeant4ProcessName
-----------------------------------------

Name of current Geant4 process 

**/
std::string CRandomEngine::CurrentGeant4ProcessName()  // static 
{
    G4VProcess* proc = CProcess::CurrentProcess() ; 
    if(proc == NULL) return "NoProc" ; 
     
    LOG(LEVEL) << "[ " << proc ; 
    const G4String& procname = proc->GetProcessName() ;  
    std::string name = procname ; 
    LOG(LEVEL) << "] (" << name << ")" ; 
    return name ; 
}

/**
CRandomEngine::FormLocation
----------------------------

Forms a location of form::

   OpBoundary_DiDiTransCoeff

**/

std::string CRandomEngine::FormLocation(const char* file, int line)
{
    assert( file ) ;  // actually the label when line is -1
    std::stringstream ss ; 
    ss << CurrentGeant4ProcessName() << "_"  ;

    if(line > -1)
    {
        std::string relpath = BFile::prefixShorten(file, "$OPTICKS_HOME/" );
        ss << relpath << "+" << line ;
    }
    else
    {
         ss << file ;  
    }
    return ss.str();
}


/**
CRandomEngine::flat_instrumented
------------------------------------

This is invoked via CG4 from a macro:: 
 
    CG4UniformRand("DiDiTransCoeff",-1)   # from DsG4OpBoundaryProcess.cc

NB not all invokations are instrumented, 
ie there are some internal calls to flat
and some external, so distinguish with m_internal

When the line argument is negative the file is regarded as a label

**/

double CRandomEngine::flat_instrumented(const char* file, int line)
{
    m_location = FormLocation(file, line) ;
    m_internal = true ; 
    double _flat = flat();
    m_internal = false ;
    return _flat ; 
}



/**
CRandomEngine::flat()
----------------------

Returns a random double in range 0..1

A StepToSmall boundary condition immediately following 
FresnelReflection is special cased to avoid calling _flat(). 
Although apparently the returned value is not used (it being from OpBoundary process)
it is still important to avoid call _flat() and misaligning the sequences.

**/

double CRandomEngine::flat() 
{ 
    if(!m_internal) m_location = CurrentGeant4ProcessName();
    assert( m_current_record_flat_count < m_seq_nv ); 


   // adding TotalInternalReflection changes history 

#ifdef USE_CUSTOM_BOUNDARY 
    bool kludge = m_dbgkludgeflatzero 
               && m_current_step_flat_count == 0
               && m_ctx._boundary_status == Ds::StepTooSmall
               && m_ctx._prior_boundary_status == Ds::FresnelReflection
               ;
#else
    bool kludge = m_dbgkludgeflatzero 
               && m_current_step_flat_count == 0
               && m_ctx._boundary_status == StepTooSmall
               && m_ctx._prior_boundary_status == FresnelReflection
               ;
#endif

#ifdef WITH_KLUDGE_FLAT_ZERO_NOPEEK
    double v = kludge ? 0.f : _flat() ; 
#else
    double v = kludge ? _peek(-2) : _flat() ; 
#endif
  
    if( kludge && m_dbgflat )
    {
        LOG(LEVEL) 
            << " --dbgkludgeflatzero  "
            << " first flat call following boundary status StepTooSmall after FresnelReflection yields  _peek(-2) or zero value "
            << " v " << v 
            ;
            // actually the value does not matter, its just OpBoundary which is not used 
    }

    m_flat = v ; 
    m_current_record_flat_count++ ;  // (*lldb*) flat 
    m_current_step_flat_count++ ; 


    if(m_dbgflat) dbgFlat(); 

    return m_flat ; 
}


int CRandomEngine::getCursor() const 
{
    return m_cursor ; 
}
int CRandomEngine::getCurrentStepFlatCount() const 
{
    return m_current_step_flat_count ; 
}
int CRandomEngine::getCurrentRecordFlatCount() const 
{
    return m_current_record_flat_count ; 
}



/*
__device__ float 
curand_uniform (curandState_t *state)
This function returns a sequence of pseudorandom floats uniformly distributed
between 0.0 and 1.0. It may return from 0.0 to 1.0, where 1.0 is included and
0.0 is excluded.

Read more at: http://docs.nvidia.com/cuda/curand/index.html
*/



double CRandomEngine::_peek(int offset) const 
{
    int idx = m_cursor + offset ; 

    bool in_range = idx >= 0 && idx < int(m_sequence.size()) ; 
    if(!in_range)
       LOG(fatal)
           << " OUT OF RANGE " 
           << " idx " << idx 
           << " m_sequence.size() " << m_sequence.size()
           ;

    assert( in_range );
    return m_sequence[idx] ; 
}


/**
CRandomEngine::_flat
----------------------

Returns random double from the canned m_sequence and increments the cursor, 
asserts that are in range.

**/

double CRandomEngine::_flat() 
{
    m_cursor += 1 ;    // m_cursor initialized to -1, and reset there by setRandomSequence, so this does start from 0  
    assert( m_cursor >= 0 && m_cursor < int(m_sequence.size()) );
    double v = m_sequence[m_cursor];
    return v  ;    
}

/**
CRandomEngine::jump
--------------------

Moves the random seqence cursor by the offset argument, either rewinding or
jumping ahead in the sequence.


**/

void CRandomEngine::jump(int offset) 
{
    m_cursor_old = m_cursor ; 
    m_jump = offset ; 
    m_jump_count += 1 ; 

    int cursor = m_cursor + offset ; 
    m_cursor = cursor ;   

    assert( cursor > -1 && cursor < int(m_sequence.size()) ) ;   // (*lldb*) jump
}


void CRandomEngine::setRandomSequence(double* s, int n) 
{
    m_sequence.clear();
    for (int i=0; i<n; i++) m_sequence.push_back(*s++);
    assert (m_sequence.size() == (unsigned)n);
    m_cursor = -1 ;
}

int CRandomEngine::findIndexOfValue(double s, double tolerance) 
{
    return SVec<double>::FindIndexOfValue(m_sequence, s, tolerance) ; 
}



void CRandomEngine::dbgFlat()
{

    const char* location = m_location.c_str() ;  

    // locseq was just for development, not needed in ordinary usage
    if(m_locseq)
    m_locseq->add(location, m_ctx._record_id, m_ctx._step_id); 

    G4VProcess* proc = CProcess::CurrentProcess() ; 
    CSteppingState ss = CStepping::CurrentState(); 

    int idx = findIndexOfValue(m_flat) ; 

    //LOG(fatal)
    std::cerr
        << " dbgFlat " 
        << desc()
        << " cur:"
        << std::setw(5) << m_cursor 
        << " idx:"
        << std::setw(5) << idx
        << " "
        << std::setw(10) << m_flat 
        << " " 
        << std::setw(20) << CStepStatus::Desc(ss.fStepStatus)
        << " " << CProcess::Desc(proc)       
        << " alignlevel " << m_alignlevel
        << std::endl 
        ; 


   if(m_dbgflatlog && m_flat > 0.)
   {
       m_dbgflatlog->addValue(location, m_flat );  
   } 

   if(m_ok->hasOpt("flatsigint") || m_curflatsigint == m_cursor ) std::raise(SIGINT);




}


void CRandomEngine::addNote(const char* note, int value)
{
    assert( m_dbgflatlog ); 
    m_dbgflatlog->addNote( note, value ); 
}

void CRandomEngine::addCut(const char* ckey, double cvalue)
{
    assert( m_dbgflatlog ); 
    m_dbgflatlog->addCut( ckey, cvalue ); 
}






/**
CRandomEngine::postStep
-------------------------

This is invoked by CG4::postStep

Normally without zeroSteps this does nothing 
other than resetting the m_current_step_flat_count to zero.

When there are zeroSteps the RNG sequence is rewound 
by -m_current_step_flat_count as if the current step never 
happened.

This rewinding for zeroSteps can be inhibited using 
the --dbgnojumpzero option. 

**/

void CRandomEngine::postStep()
{


/*
    if(m_ctx._noZeroSteps > 0)
    {
        int backseq = -m_current_step_flat_count ; 
        bool dbgnojumpzero = m_ok->isDbgNoJumpZero() ; 

        if(m_dbgflat)
            LOG(LEVEL) 
              << " _noZeroSteps " << m_ctx._noZeroSteps
              << " backseq " << backseq
              << " --dbgnojumpzero " << ( dbgnojumpzero ? "YES" : "NO" )
              ;

        if( dbgnojumpzero )
        {
            LOG(debug) << "rewind inhibited by option: --dbgnojumpzero " ;   
        }
        else
        {
            jump(backseq);
        }
    }
*/


    if(m_masked)
    {
        std::string seq = OpticksFlags::FlagSequence(m_okevt_seqhis, true, m_ctx._step_id_valid + 1  );
        m_okevt_pt = strdup(seq.c_str()) ;
        LOG(debug) 
           << " m_ctx._record_id:  " << m_ctx._record_id 
           << " ( m_okevt_seqhis: " << std::hex << m_okevt_seqhis << std::dec
           << " okevt_pt " << m_okevt_pt  << " ) "
           ;
    }

    m_current_step_flat_count = 0 ;   // (*lldb*) postStep 

    m_step_cursors.push_back(m_cursor) ; 


    if( m_locseq )
    {
        m_locseq->postStep();
        LOG(info) << CProcessManager::Desc(m_ctx._process_manager) ; 
    }
}






/**
CRandomEngine::preTrack
-------------------------

Invoked from CG4::preTrack following CCtx::setTrack


Masked Running 
~~~~~~~~~~~~~~~~~~

Use of the maskIndex allows a partial run on a selection of  
photons to use the same RNG sequence as a full run 
over many photons, buy jumping to the appropriate item line 
RNG sequence.


Masked Running Currently Limited to input photons 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hmm this aint going to work for aligning generation 
as the track aint born yet... need to set the photon
index back in the PostStepDoIt generation loop/ 

Currently are avoiding this issue by only using alignment
with input photons.

And need to retain separate cursors for the streams for 
each photon, as the generation loop generates all photons
each consuming a varying number of RNG  
and then starts propagating them.  This differs from Opticks
which does generation and propagation in each thread sequentially.

**/

void CRandomEngine::preTrack()
{
    m_jump = 0 ; 
    m_jump_count = 0 ; 

    bool has_mask = m_ok->hasMask(); 
    unsigned use_index = has_mask ? m_ctx._mask_index : m_ctx._record_id ; 

    setupCurandSequence(use_index) ;   
   
    unsigned mask_size = m_ok->getMaskSize() ; 
    if(mask_size == 1)
    {
        run_ucf_script( use_index ) ; 
        m_pindexlogpath = PindexLogPath(use_index) ;   
        m_pindexlog = BLog::Load(m_pindexlogpath); 
        m_dbgflatlog = new BLog ; 
        //m_pindexlog->dump("CRandomEngine::preTrack.pindexlog"); 
        dumpPindexLog("CRandomEngine::preTrack.dumpPindexLog"); 
    }

    LOG(debug)
        << "record_id: "    // (*lldb*) preTrack
        << " ctx._record_id " << m_ctx._record_id 
        << " ctx._mask_index " << m_ctx._mask_index
        << " use_index " << use_index 
        << " has_mask " << ( has_mask ? "YES" : "NO" )
        << " mask_size " << mask_size 
        << " pindexlogpath " << m_pindexlogpath 
        ;
}

const char* CRandomEngine::PindexLogPath(unsigned mask_index)
{
    return BStr::concat<unsigned>("$TMP/ox_", mask_index, ".log" );  
}

void CRandomEngine::dumpPindexLog(const char* msg)
{
    LOG(info) << msg ; 
    BLog* log = m_pindexlog ; 
    log->setSequence(&m_sequence); 
    log->dump("CRandomEngine::dumpPindexLog"); 
}

/**
CRandomEngine::compareLogs
----------------------------

Invoked from CRandomEngine::postTrack when --dbgflat is used.

**/

void CRandomEngine::compareLogs(const char* msg)    // --dbgflat
{
    LOG(info) << msg ; 
    BLog* a = m_pindexlog ; 
    BLog* b = m_dbgflatlog ; 

    a->setSequence(&m_sequence) ;  
    b->setSequence(&m_sequence) ;  

    b->dump("CRandomEngine::compareLogs:B --dbgflat "); 

    int RC = BLog::Compare( a, b );  
    LOG(info) << msg << " RC " << RC ; 
}




/**
CRandomEngine::run_ucf_script
----------------------------------

Access to the Opticks event, relies on Opticks GPU propagation
going first, which is only the case in align mode.

**/


void CRandomEngine::run_ucf_script(unsigned mask_index) 
{
    assert(m_run); 
    if(!m_okevt) m_okevt = m_run->getEvent();  // is the defer needed ?
    m_okevt_seqhis  = m_okevt ? m_okevt->getSeqHis(m_ctx._record_id ) : 0ull ;

    LOG(info) 
       << " [ --align --mask ] " 
       << " m_ctx._record_id:  " << m_ctx._record_id 
       << " mask_index: " << mask_index 
       << " ( m_okevt_seqhis: " << std::hex << m_okevt_seqhis << std::dec
       << " " << OpticksFlags::FlagSequence(m_okevt_seqhis) << " ) "
       ;

    const char* cmd = BStr::concat<unsigned>("ucf.py ", mask_index, NULL );  
    LOG(info) << "[ cmd \"" << cmd << "\"";   
    int rc = SSys::run(cmd) ;  // NB must not write to stdout, stderr is ok though 
    assert( rc == 0 );
    LOG(info) << "] cmd \"" << cmd << "\"";   
}


void CRandomEngine::postTrack()
{
    if(m_jump_count > 0)
    {
        m_jump_photons.push_back(m_ctx._record_id);
    }


    if(m_locseq)   // (*lldb*) postTrack
    {
        unsigned long long seqhis = m_manager->getSeqHis()  ;
        m_locseq->mark(seqhis);
        LOG(info) << CProcessManager::Desc(m_ctx._process_manager) ; 
    }

    if(m_dbgflat)
    {
        compareLogs("CRandomEngine::postTrack");  
    }

}

void CRandomEngine::dump(const char* msg) const 
{
    if(!m_locseq) return ; 
    m_locseq->dump(msg);
}


const char* CRandomEngine::TMPDIR = "$TMP/cfg4/CRandomEngine"  ; 

void CRandomEngine::postpropagate()
{
    LOG(info) 
        << " jump_photons " << m_jump_photons.size()
        ;

    NPY<unsigned>* jump = NPY<unsigned>::make_from_vec(m_jump_photons) ;
    jump->save(TMPDIR, "CRandomEngine_jump_photons.npy");


    dump("CRandomEngine::postpropagate");
}






void CRandomEngine::flatArray(const int, double* ) 
{
    assert(0);
}
void CRandomEngine::setSeed(long, int ) 
{
    assert(0);
} 
void CRandomEngine::setSeeds(const long *, int) 
{
    assert(0);
}
void CRandomEngine::saveStatus( const char * ) const 
{
    assert(0);
}       
void CRandomEngine::restoreStatus( const char * )
{
    assert(0);
}
void CRandomEngine::showStatus() const 
{
    assert(0);
}


