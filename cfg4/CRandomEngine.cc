
#include <iomanip>

#include "PLOG.hh"

#include "Randomize.hh"
#include "G4String.hh"
#include "G4VProcess.hh"


#include "SSys.hh"

#include "BStr.hh"
#include "BFile.hh"
#include "BLocSeq.hh"

#include "Opticks.hh"
#include "OpticksRun.hh"
#include "OpticksEvent.hh"
#include "OpticksFlags.hh"

#include "NPY.hpp"

#include "CG4.hh"
#include "CProcess.hh"
#include "CProcessManager.hh"
#include "CStepStatus.hh"
#include "CStepping.hh"
#include "CRandomEngine.hh"



std::string CRandomEngine::name() const 
{
    return "CRandomEngine";
}

CRandomEngine::CRandomEngine(CG4* g4)
    :
    m_g4(g4),
    m_ctx(g4->getCtx()),
    m_ok(g4->getOpticks()),
    m_dbgkludgeflatzero(m_ok->isDbgKludgeFlatZero()), 
    m_run(g4->getRun()),
    m_okevt(NULL),
    m_okevt_seqhis(0),
    m_okevt_pt(NULL),
    m_g4evt(NULL),
    m_mask(m_ok->getMask()),
    m_masked(m_mask.size() > 0),
    m_path("$TMP/TRngBufTest.npy"),
    m_alignlevel(m_ok->getAlignLevel()),
    m_seed(9876),
    m_internal(false),
    m_skipdupe(true),
    m_locseq(m_alignlevel > 1 ? new BLocSeq<unsigned long long>(m_skipdupe) : NULL ),
    m_curand(NPY<double>::load(m_path)),
    m_curand_index(-1),
    m_curand_ni(m_curand ? m_curand->getShape(0) : 0 ),
    m_curand_nv(m_curand ? m_curand->getNumValues(1) : 0 ),
    m_current_record_flat_count(0),
    m_current_step_flat_count(0),
    m_jump(0),
    m_jump_count(0),
    m_flat(-1.0),
    m_cursor(0),
    m_cursor_old(0)
{
    init();
}

bool CRandomEngine::hasSequence() const 
{
    return m_curand && m_curand_ni > 0 && m_curand_nv > 0 ; 
}

const char* CRandomEngine::getPath() const 
{
    return m_path ; 
}

void CRandomEngine::dumpDouble(const char* msg, double* v, unsigned width ) const 
{
    LOG(info) << msg ; 
    assert( m_curand_nv > 15 );
    for(int i=0 ; i < 16 ; i++)  
    {
        std::cout << std::fixed << std::setw(10) << std::setprecision(10) << v[i] << " " ; 
        if( i % width == (width - 1) ) std::cout << std::endl ; 
    }
}

void CRandomEngine::init()
{
    initCurand();
    CLHEP::HepRandom::setTheEngine( this );  
}

void CRandomEngine::initCurand()
{
    LOG(info) << ( m_curand ? m_curand->getShapeString() : "-" ) 
              << " curand_ni " << m_curand_ni
              << " curand_nv " << m_curand_nv
              ; 

    if(!m_curand) return ; 
        
    unsigned w = 4 ; 
    if( m_curand_ni > 0 )
         dumpDouble( "v0" , m_curand->getValues(0), w ) ; 

    if( m_curand_ni > 1 )
         dumpDouble( "v1" , m_curand->getValues(1), w ) ; 

    if( m_curand_ni > 99999 )
        dumpDouble( "v99999" , m_curand->getValues(99999), w ) ; 
}

void CRandomEngine::setupCurandSequence(int record_id)
{
    if( m_curand_ni == 0 )
    {
        LOG(fatal) << "CRandomEngine::setupCurandSequence"
                   << " m_curand_ni ZERO "
                   << " no precooked RNG have been loaded from " 
                   << " m_path " << m_path
                   << " : try running : TRngBufTest "
                   ;

    }
    assert( m_curand_ni > 0 );

    assert( record_id > -1 && record_id < m_curand_ni ); 

    assert( m_curand_nv > 0 ) ;

    m_curand_index = record_id ; 

    double* seq = m_curand->getValues(record_id) ; 

    setRandomSequence( seq, m_curand_nv ) ; 

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


std::string CRandomEngine::CurrentProcessName()
{
    G4VProcess* proc = CProcess::CurrentProcess() ; 
    std::stringstream ss ; 
    ss <<  ( proc ? proc->GetProcessName().c_str() : "NoProc" )  ;  
    return ss.str();
}

std::string CRandomEngine::FormLocation(const char* file, int line)
{
    assert( file ) ;  // actually the label when line is -1
    std::stringstream ss ; 
    ss << CurrentProcessName() << "_"  ;

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


// NB not all invokations are instrumented, 
//    ie there are some internal calls to flat
//    and some external, so distinguish with m_internal

double CRandomEngine::flat_instrumented(const char* file, int line)
{
    // when line is negative the file is regarded as a label
    m_location = FormLocation(file, line) ;
    m_internal = true ; 
    double _flat = flat();
    m_internal = false ;
    return _flat ; 
}


double CRandomEngine::flat() 
{ 
    if(!m_internal) m_location = CurrentProcessName();
    assert( m_current_record_flat_count < m_curand_nv ); 
 
    bool kludge = m_dbgkludgeflatzero 
               && m_current_step_flat_count == 0
               && m_ctx._boundary_status == StepTooSmall
               && m_ctx._prior_boundary_status == FresnelReflection 
               ;

    double v = kludge ? _peek(-3) : _flat() ; 
  
    if( kludge )
    {
        LOG(info) << " --dbgkludgeflatzero  "
                  << " first flat call following FresnelReflection then StepTooSmall yields  _peek(-3) value "
                  << " v " << v 
                 ;
    }

    m_flat = v ; 

    m_current_record_flat_count++ ;  // (*lldb*) flat 
    m_current_step_flat_count++ ; 

    return m_flat ; 
}


double CRandomEngine::_peek(int offset) const 
{
     unsigned idx = m_cursor + offset ; 
     assert( idx < m_sequence.size() );
     return m_sequence[idx] ; 
}

double CRandomEngine::_flat() 
{
    assert( m_cursor < m_sequence.size() );
    double v = m_sequence[m_cursor];
    m_cursor += 1 ; 
    return v  ;    
}

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
    m_cursor = 0;
}

void CRandomEngine::dumpFlat()
{
    // locseq was just for development, not needed in ordinary usage
    if(m_locseq)
    m_locseq->add(m_location.c_str(), m_ctx._record_id, m_ctx._step_id); 

    G4VProcess* proc = CProcess::CurrentProcess() ; 
    CSteppingState ss = CStepping::CurrentState(); 
    std::cerr 
        << desc()
        << " "
        << std::setw(10) << m_flat 
        << " " 
        << std::setw(20) << CStepStatus::Desc(ss.fStepStatus)
        << " " << CProcess::Desc(proc)       
        << " alignlevel " << m_alignlevel
        <<  std::endl 
        ; 
}


// invoked by CG4::postStep
void CRandomEngine::postStep()
{
    if(m_ctx._noZeroSteps > 0)
    {
        //int backseq = -2*m_current_step_flat_count ; 
        int backseq = -m_current_step_flat_count ; 
        bool dbgnojumpzero = m_ok->isDbgNoJumpZero() ; 

        LOG(error) << "CRandomEngine::postStep"
                   << " _noZeroSteps " << m_ctx._noZeroSteps
                   << " backseq " << backseq
                   << " --dbgnojumpzero " << ( dbgnojumpzero ? "YES" : "NO" )
                   ;

        if( dbgnojumpzero )
        {
            LOG(fatal) << "CRandomEngine::postStep rewind inhibited by option: --dbgnojumpzero " ;   
        }
        else
        {
            jump(backseq);
        }
    }


    if(m_masked)
    {
        m_okevt_pt = OpticksFlags::PointAbbrev(m_okevt_seqhis, m_ctx._step_id + 1  ) ;
        LOG(debug) 
           << " m_ctx._record_id:  " << m_ctx._record_id 
           << " ( m_okevt_seqhis: " << std::hex << m_okevt_seqhis << std::dec
           << " okevt_pt " << m_okevt_pt  << " ) "
           ;
    }

    m_current_step_flat_count = 0 ;   // (*lldb*) postStep 

    if( m_locseq )
    {
        m_locseq->postStep();
        LOG(info) << CProcessManager::Desc(m_ctx._process_manager) ; 
    }
}


// invoked from CG4::preTrack following CG4Ctx::setTrack
void CRandomEngine::preTrack()
{
    m_jump = 0 ; 
    m_jump_count = 0 ; 


    unsigned use_index ; 
    bool with_mask = m_ok->hasMask();

    if(with_mask)
    {
        unsigned mask_index = m_ok->getMaskIndex( m_ctx._record_id ); 
        use_index = mask_index ; 

        if(!m_okevt) m_okevt = m_run->getEvent();  // is the defer needed ?
        m_okevt_seqhis  = m_okevt ? m_okevt->getSeqHis(m_ctx._record_id ) : 0ull ;

        LOG(info) 
           << " m_ctx._record_id:  " << m_ctx._record_id 
           << " mask_index: " << mask_index 
           << " ( m_okevt_seqhis: " << std::hex << m_okevt_seqhis << std::dec
           << " " << OpticksFlags::FlagSequence(m_okevt_seqhis) << " ) "
           ;

        const char* cmd = BStr::concat<unsigned>("ucf.py ", mask_index, NULL );  
        LOG(info) << "CRandomEngine::preTrack : START cmd \"" << cmd << "\"";   
        int rc = SSys::run(cmd) ;  // NB must not write to stdout, stderr is ok though 
        assert( rc == 0 );
        LOG(info) << "CRandomEngine::preTrack : DONE cmd \"" << cmd << "\"";   
    }
    else
    {
        use_index = m_ctx._record_id ;
    }

    setupCurandSequence(use_index) ;   

    LOG(error) << "CRandomEngine::pretrack record_id: "    // (*lldb*) preTrack
               << " ctx.record_id " << m_ctx._record_id 
               << " use_index " << use_index 
               << " with_mask " << ( with_mask ? "YES" : "NO" )
               ;
 
}


void CRandomEngine::postTrack()
{
    if(m_jump_count > 0)
    {
        m_jump_photons.push_back(m_ctx._record_id);
    }


    if(m_locseq)   // (*lldb*) postTrack
    {
        unsigned long long seqhis = m_g4->getSeqHis()  ;
        m_locseq->mark(seqhis);
        LOG(info) << CProcessManager::Desc(m_ctx._process_manager) ; 
    }
}

void CRandomEngine::dump(const char* msg) const 
{

    


    if(!m_locseq) return ; 
    m_locseq->dump(msg);
}

void CRandomEngine::postpropagate()
{

    LOG(info) << "CRandomEngine::postpropagate"
              << " jump_photons " << m_jump_photons.size()
              ;

    NPY<unsigned>* jump = NPY<unsigned>::make_from_vec(m_jump_photons) ;
    jump->save("$TMP/CRandomEngine_jump_photons.npy");


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


