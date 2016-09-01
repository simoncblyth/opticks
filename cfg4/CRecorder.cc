#include "CFG4_BODY.hh"
// brap-
#include "BBit.hh"

// npy-
#include "NGLM.hpp"
#include "NPY.hpp"

// okc-
#include "Opticks.hh"
#include "OpticksFlags.h"
#include "OpticksFlags.hh"
#include "OpticksEvent.hh"

// okg-
#include "OpticksHub.hh"

// g4-
#include "G4Track.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4StepStatus.hh"
#include "G4ThreeVector.hh"
#include "G4PrimaryVertex.hh"
#include "G4PrimaryParticle.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

// cfg4-
#include "OpStatus.hh"
#include "CRecorder.h"
#include "CPropLib.hh"
#include "Format.hh"
#include "CRecorder.hh"

#include "PLOG.hh"


#define fitsInShort(x) !(((((x) & 0xffff8000) >> 15) + 1) & 0x1fffe)
#define iround(x) ((x)>=0?(int)((x)+0.5):(int)((x)-0.5))

short shortnorm( float v, float center, float extent )
{
    // range of short is -32768 to 32767
    // Expect no positions out of range, as constrained by the geometry are bouncing on,
    // but getting times beyond the range eg 0.:100 ns is expected
    //  
    int inorm = iround(32767.0f * (v - center)/extent ) ;    // linear scaling into -1.f:1.f * float(SHRT_MAX)
    return fitsInShort(inorm) ? short(inorm) : SHRT_MIN  ;
} 

unsigned char uchar_( float f )  // f in range -1.:1. 
{
    int ipol = iround((f+1.f)*127.f) ;
    return ipol ; 
}

const char* CRecorder::PRE  = "PRE" ; 
const char* CRecorder::POST = "POST" ; 




CRecorder::CRecorder(OpticksHub* hub, CPropLib* clib, bool dynamic) 
   :
   m_hub(hub),
   m_evt(NULL),
   m_clib(clib),
   m_dynamic(dynamic),
   m_gen(0),

   m_record_max(0),
   m_bounce_max(0),
   m_steps_per_photon(0), 

   m_photons_per_g4event(0),

   m_verbosity(m_hub->hasOpt("steppingdbg") ? 10 : 0),
   m_debug(m_verbosity > 0),
   m_event_id(UINT_MAX),
   m_photon_id(UINT_MAX),
   m_step_id(UINT_MAX),
   m_record_id(UINT_MAX),

   m_primary_id(UINT_MAX),
   m_primary_max(0),

   m_boundary_status(Undefined),
   m_prior_boundary_status(Undefined),

   m_premat(0),
   m_prior_premat(0),

   m_postmat(0),
   m_prior_postmat(0),

   m_seqhis(0),
   m_seqmat(0),
   m_seqhis_select(0),
   m_seqmat_select(0),
   m_slot(0),
   m_truncate(false),
   m_step(true),

   m_primary(0),
   m_photons(0),
   m_records(0),
   m_history(0),


   m_dynamic_primary(NULL),
   m_dynamic_records(NULL),
   m_dynamic_photons(NULL),
   m_dynamic_history(NULL)
{
   init();
   
}

void CRecorder::init()
{
}



unsigned int CRecorder::getVerbosity()
{
    return m_verbosity ; 
}
bool CRecorder::isHistorySelected()
{
   return m_seqhis_select == m_seqhis ; 
}
bool CRecorder::isMaterialSelected()
{
   return m_seqmat_select == m_seqmat ; 
}
bool CRecorder::isSelected()
{
   return isHistorySelected() || isMaterialSelected() ;
}

unsigned long long CRecorder::getSeqHis()
{
    return m_seqhis ; 
}
unsigned long long CRecorder::getSeqMat()
{
    return m_seqmat ; 
}






void CRecorder::setPropLib(CPropLib* clib)
{
    m_clib = clib  ; 
}


//OpticksEvent* CRecorder::getEvent()
//{
//    return m_evt ; 
//}

unsigned int CRecorder::getRecordMax()
{
    return m_record_max ; 
}


unsigned int CRecorder::getEventId()
{
   return m_event_id ; 
}
unsigned int CRecorder::getPhotonId()
{
   return m_photon_id ; 
}
unsigned int CRecorder::getStepId()
{
   return m_step_id ; 
}
unsigned int CRecorder::getRecordId()
{
   return m_record_id ; 
}




G4OpBoundaryProcessStatus CRecorder::getBoundaryStatus()
{
   return m_boundary_status ; 
}



void CRecorder::setEventId(unsigned int event_id)
{
    m_event_id = event_id ; 
}
void CRecorder::setPhotonId(unsigned int photon_id)
{
    m_photon_id = photon_id ; 
}
void CRecorder::setStepId(unsigned int step_id)
{
    m_step_id = step_id ; 
}
unsigned int CRecorder::defineRecordId()   
{
   return m_photons_per_g4event*m_event_id + m_photon_id ; 
}

void CRecorder::setRecordId(unsigned int record_id)
{
    m_record_id = record_id ; 
}



void CRecorder::RecordBeginOfRun(const G4Run*)
{
}

void CRecorder::RecordEndOfRun(const G4Run*)
{
}


//bool CRecorder::isDynamic()
//{
//    return m_dynamic ; 
//}



void CRecorder::initEvent()
{
    m_evt = m_hub->getG4Event();

    m_c4.u = 0u ; 

    m_photons_per_g4event = m_evt->getNumPhotonsPerG4Event() ; 
    m_record_max = m_evt->getNumPhotons();   // from the genstep summation

    m_bounce_max = m_evt->getBounceMax();
    m_steps_per_photon = m_evt->getMaxRec() ;    

    if(m_dynamic)
    {
        assert(m_record_max == 0 );

        // shapes must match OpticksEvent::createBuffers
        // TODO: avoid this duplicity 

        m_dynamic_records = NPY<short>::make(1, m_steps_per_photon, 2, 4) ;
        m_dynamic_records->zero();

        m_dynamic_photons = NPY<float>::make(1, 4, 4) ;
        m_dynamic_photons->zero();

        m_dynamic_primary = NPY<float>::make(1, 4, 4) ;
        m_dynamic_primary->zero();

        m_dynamic_history = NPY<unsigned long long>::make(1, 1, 2) ;
        m_dynamic_history->zero();

    } 
    else
    {
        assert(m_record_max > 0 );
    }

    //m_step = m_evt->isStep();
    m_step = true ;

    LOG(info) << "CRecorder::initEvent"
              << " dynamic " << ( m_dynamic ? "DYNAMIC(CPU style)" : "STATIC(GPU style)" )
              << " record_max " << m_record_max
              << " bounce_max  " << m_bounce_max 
              << " steps_per_photon " << m_steps_per_photon 
              << " photons_per_g4event " << m_photons_per_g4event
              << " num_g4event " << m_evt->getNumG4Event() 
              << " isStep " << m_step  
              ;

    //m_evt->zero();

    m_history = m_evt->getSequenceData();
    m_photons = m_evt->getPhotonData();
    m_records = m_evt->getRecordData();

    assert( m_history && "CRecorder requires history buffer" );
    assert( m_photons && "CRecorder requires photons buffer" );
    assert( m_records && "CRecorder requires records buffer" );

    const char* typ = m_evt->getTyp();

    m_gen = OpticksFlags::SourceCode(typ);

    assert( m_gen == TORCH || m_gen == G4GUN  );



}



void CRecorder::startPhoton()
{
    //LOG(trace) << "CRecorder::startPhoton" ; 

    if(m_record_id % 10000 == 0) Summary("CRecorder::startPhoton(%10k)") ;

    assert(m_step_id == 0);

    m_c4.u = 0u ; 

    m_boundary_status = Undefined ; 
    m_prior_boundary_status = Undefined ; 

    m_premat = 0 ; 
    m_prior_premat = 0 ; 

    m_postmat = 0 ; 
    m_prior_postmat = 0 ; 

    m_seqmat = 0 ; 
    m_seqhis = 0 ; 

    //m_seqhis_select = 0xfbbbbbbbcd ;
    //m_seqhis_select = 0x8cbbbbbc0 ;
    m_seqhis_select = 0x8bd ;

    m_slot = 0 ; 
    m_truncate = false ; 

    if(m_debug) Clear();
}


void CRecorder::setBoundaryStatus(G4OpBoundaryProcessStatus boundary_status, unsigned int premat, unsigned int postmat)
{
    // this is invoked before RecordStep is called from SteppingAction
    m_prior_boundary_status = m_boundary_status ; 
    m_prior_premat = m_premat ; 
    m_prior_postmat = m_postmat ; 

    m_boundary_status = boundary_status ; 
    m_premat = premat ; 
    m_postmat = postmat ; 
}
  
bool CRecorder::RecordStep(const G4Step* step)
{
    const G4StepPoint* pre  = step->GetPreStepPoint() ; 
    const G4StepPoint* post = step->GetPostStepPoint() ; 

    unsigned int preFlag ; 
    unsigned int postFlag ; 
    unsigned int preMat ; 
    unsigned int postMat ; 

    // shift flags by 1 relative to steps, in order to set the generation code on first step
    // this doesnt miss flags, as record both pre and post at last step    

    if(m_step_id == 0)
    {
        preFlag = m_gen ;         
        postFlag = OpPointFlag(post, m_boundary_status) ;
    }
    else
    {
        preFlag  = OpPointFlag(pre,  m_prior_boundary_status);
        postFlag = OpPointFlag(post, m_boundary_status) ;
    }

    bool lastPost = (postFlag & (BULK_ABSORB | SURFACE_ABSORB | SURFACE_DETECT)) != 0 ;
    bool surfaceAbsorb = (postFlag & (SURFACE_ABSORB | SURFACE_DETECT)) != 0 ;

    bool preSkip = m_prior_boundary_status == StepTooSmall ;  

    bool matSwap = m_boundary_status == StepTooSmall ; 

    preMat  = matSwap ? m_postmat : m_premat ;
    postMat = ( matSwap || m_postmat == 0 || surfaceAbsorb )  ? m_premat  : m_postmat ;

    bool done = false ; 

    // skip the pre, but the post becomes the pre at next step where will be taken 
    // 1-based material indices, so zero can represent None
    if(!preSkip)
    {
       done = RecordStepPoint( pre, preFlag, preMat, m_prior_boundary_status, PRE ); 
    }

    if(lastPost && !done)
    {
       done = RecordStepPoint( post, postFlag, postMat, m_boundary_status, POST ); 
    }

    // when not *absorb* the post step will become the pre step at next RecordStep

    return done ;
}


bool CRecorder::RecordStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, G4OpBoundaryProcessStatus boundary_status, const char* label)
{
    bool absorb = ( flag & (BULK_ABSORB | SURFACE_ABSORB | SURFACE_DETECT)) != 0 ;

    unsigned int slot =  m_slot < m_steps_per_photon  ? m_slot : m_steps_per_photon - 1 ;

    //Dump(label,  slot, point, boundary_status );

    if(m_step)
    {
        unsigned long long shift = slot*4ull ;   
        unsigned long long msk = 0xFull << shift ; 
        unsigned long long his = BBit::ffs(flag) & 0xFull ; 
        unsigned long long mat = material < 0xFull ? material : 0xFull ; 
        m_seqhis =  (m_seqhis & (~msk)) | (his << shift) ; 
        m_seqmat =  (m_seqmat & (~msk)) | (mat << shift) ; 

        RecordStepPoint(slot, point, flag, material, label);

        if(m_debug) Collect(point, flag, material, boundary_status, m_seqhis, m_seqmat);
    }

    m_slot += 1 ; 

    bool truncate = m_slot > m_bounce_max  ;  
    bool done = truncate || absorb ;

    if(done && m_dynamic)
    {
        m_records->add(m_dynamic_records);
    }

    return done ; 
}


void CRecorder::RecordStepPoint(unsigned int slot, const G4StepPoint* point, unsigned int flag, unsigned int material, const char* /*label*/ )
{

/*
    LOG(trace) << "CRecorder::RecordStepPoint" 
              << " m_record_id " << m_record_id 
              << " m_step_id " << m_step_id 
              << " m_slot " << m_slot 
              << " slot " << slot 
              << " flag " << flag
              << " m_seqhis " << std::hex << m_seqhis << std::dec 
              ;

*/

    const G4ThreeVector& pos = point->GetPosition();
    //const G4ThreeVector& dir = point->GetMomentumDirection();
    const G4ThreeVector& pol = point->GetPolarization();

    G4double time = point->GetGlobalTime();
    G4double energy = point->GetKineticEnergy();
    G4double wavelength = h_Planck*c_light/energy ;
    //G4double weight = 1.0 ; 

    const glm::vec4& sd = m_evt->getSpaceDomain() ; 
    const glm::vec4& td = m_evt->getTimeDomain() ; 
    const glm::vec4& wd = m_evt->getWavelengthDomain() ; 

    short posx = shortnorm(pos.x()/mm, sd.x, sd.w ); 
    short posy = shortnorm(pos.y()/mm, sd.y, sd.w ); 
    short posz = shortnorm(pos.z()/mm, sd.z, sd.w ); 
    short time_ = shortnorm(time/ns,   td.x, td.y );

    /*
    LOG(info) << "CRecorder::RecordStepPoint"
              << " globalTime " << time 
              << " td.x " << td.x
              << " td.y " << td.y
              << " ns " << ns
              << " time/ns " << time/ns
              <<  " time_ " << time_
              ;
     */
  


    unsigned char polx = uchar_( pol.x() );
    unsigned char poly = uchar_( pol.y() );
    unsigned char polz = uchar_( pol.z() );
    unsigned char wavl = uchar_( 255.f*(wavelength/nm - wd.x)/wd.w );

    qquad qaux ; 
    qaux.uchar_.x = material ; 
    qaux.uchar_.y = 0 ; // TODO:m2 
    qaux.char_.z  = 0 ; // TODO:boundary (G4 equivalent ?)
    qaux.uchar_.w = BBit::ffs(flag) ;   // ? duplicates seqhis  

    hquad polw ; 
    polw.ushort_.x = polx | poly << 8 ; 
    polw.ushort_.y = polz | wavl << 8 ; 
    polw.ushort_.z = qaux.uchar_.x | qaux.uchar_.y << 8  ;
    polw.ushort_.w = qaux.uchar_.z | qaux.uchar_.w << 8  ;

    NPY<short>* target = m_dynamic ? m_dynamic_records : m_records ; 
    unsigned int target_record_id = m_dynamic ? 0 : m_record_id ; 

    target->setQuad(target_record_id, slot, 0, posx, posy, posz, time_ );
    target->setQuad(target_record_id, slot, 1, polw.short_.x, polw.short_.y, polw.short_.z, polw.short_.w );  

    // dynamic mode : fills in slots into single photon dynamic_records structure 
    // static mode  : fills directly into a large fixed dimension records structure
}

void CRecorder::RecordQuadrant(const G4Step* step)
{
    const G4StepPoint* pre  = step->GetPreStepPoint() ; 
    const G4ThreeVector& pos = pre->GetPosition();

    // initial quadrant 
    m_c4.uchar_.x = 
                  (  pos.x() > 0.f ? unsigned(QX) : 0u ) 
                   |   
                  (  pos.y() > 0.f ? unsigned(QY) : 0u ) 
                   |   
                  (  pos.z() > 0.f ? unsigned(QZ) : 0u )
                  ;   

    m_c4.uchar_.y = 2u ; 
    m_c4.uchar_.z = 3u ; 
    m_c4.uchar_.w = 4u ; 
}

void CRecorder::RecordPhoton(const G4Step* step)
{
    // gets called at last step (eg absorption) or when truncated

    const G4StepPoint* point  = step->GetPostStepPoint() ; 

    const G4ThreeVector& pos = point->GetPosition();
    const G4ThreeVector& dir = point->GetMomentumDirection();
    const G4ThreeVector& pol = point->GetPolarization();

    G4double time = point->GetGlobalTime();
    G4double energy = point->GetKineticEnergy();
    G4double wavelength = h_Planck*c_light/energy ;
    G4double weight = 1.0 ; 

    NPY<float>* target = m_dynamic ? m_dynamic_photons : m_photons ; 
    unsigned int target_record_id = m_dynamic ? 0 : m_record_id ; 


    target->setQuad(target_record_id, 0, 0, pos.x()/mm, pos.y()/mm, pos.z()/mm, time/ns  );
    target->setQuad(target_record_id, 1, 0, dir.x(), dir.y(), dir.z(), weight  );
    target->setQuad(target_record_id, 2, 0, pol.x(), pol.y(), pol.z(), wavelength/nm  );

    target->setUInt(target_record_id, 3, 0, 0, m_slot );
    target->setUInt(target_record_id, 3, 0, 1, 0u );
    target->setUInt(target_record_id, 3, 0, 2, m_c4.u );
    target->setUInt(target_record_id, 3, 0, 3, 0u );

    if(m_dynamic)
    {
        m_photons->add(m_dynamic_photons);
    }

    // generate.cu
    //
    //  (x)  p.flags.i.x = prd.boundary ;   // last boundary
    //  (y)  p.flags.u.y = s.identity.w ;   // sensorIndex  >0 only for cathode hits
    //  (z)  p.flags.u.z = s.index.x ;      // material1 index  : redundant with boundary  
    //  (w)  p.flags.u.w |= s.flag ;        // OR of step flags : redundant ? unless want to try to live without seqhis
    //

    if(m_step)
    {

        NPY<unsigned long long>* h_target = m_dynamic ? m_dynamic_history : m_history ; 

        unsigned long long* history = h_target->getValues() + 2*target_record_id ;
        *(history+0) = m_seqhis ; 
        *(history+1) = m_seqmat ; 

        if(m_dynamic)
        {
            m_history->add(m_dynamic_history);
        }
    }
}

bool CRecorder::hasIssue()
{
    unsigned int npoints = m_points.size() ;
    assert(m_flags.size() == npoints);
    assert(m_materials.size() == npoints);
    assert(m_bndstats.size() == npoints);

    bool issue = false ; 
    for(unsigned int i=0 ; i < npoints ; i++) 
    {
       if(m_flags[i] == 0 || m_flags[i] == NAN_ABORT) issue = true ; 
    }
    return issue ; 
}

void CRecorder::Dump(const char* msg, unsigned int index, const G4StepPoint* point, G4OpBoundaryProcessStatus boundary_status, const char* matname )
{
    LOG(info) << msg ; 
    std::string bs = OpBoundaryAbbrevString(boundary_status) ;
    G4ThreeVector origin ; 
    std::cout << std::setw(7) << index << " " << std::setw(15) << matname << " " << Format(point, origin, bs.c_str()) << std::endl ;
}

void CRecorder::Dump(const char* msg)
{
    LOG(info) << msg 
              << " record_id " << std::setw(7) << m_record_id 
              ;
    LOG(info) 
              << " seqhis " << std::hex << m_seqhis << std::dec 
              << " " << OpticksFlags::FlagSequence(m_seqhis) 
              ;
    LOG(info) 
              << " seqmat " << std::hex << m_seqmat << std::dec 
              << " " << m_clib->MaterialSequence(m_seqmat) 
              ;

    if(!m_debug) return ; 

    for(unsigned int i=0 ; i<m_points.size() ; i++) 
    {
       //unsigned long long seqhis = m_seqhis_dbg[i] ;
       //unsigned long long seqmat = m_seqmat_dbg[i] ;
       G4OpBoundaryProcessStatus bst = m_bndstats[i] ;
       unsigned int mat = m_materials[i] ;
       const char* matname = ( mat == 0 ? "-" : m_clib->getMaterialName(mat-1)  ) ;

       Dump(msg, i, m_points[i], bst, matname );

       //std::cout << std::hex << seqhis << std::dec << std::endl ; 
       //std::cout << std::hex << seqmat << std::dec << std::endl ; 
    }
}

void CRecorder::Collect(const G4StepPoint* point, unsigned int flag, unsigned int material, G4OpBoundaryProcessStatus boundary_status, unsigned long long seqhis, unsigned long long seqmat)
{
    assert(m_debug);
    m_points.push_back(new G4StepPoint(*point));
    m_flags.push_back(flag);
    m_materials.push_back(material);
    m_bndstats.push_back(boundary_status);  // will duplicate the status for the last step
    m_seqhis_dbg.push_back(seqhis);
    m_seqmat_dbg.push_back(seqmat);
}

void CRecorder::Clear()
{
    assert(m_debug);
    for(unsigned int i=0 ; i < m_points.size() ; i++) delete m_points[i] ;
    m_points.clear();
    m_flags.clear();
    m_materials.clear();
    m_bndstats.clear();
    m_seqhis_dbg.clear();
    m_seqmat_dbg.clear();
}


/*
void CRecorder::setupPrimaryRecording()
{
    m_evt->prepareForPrimaryRecording();

    m_primary = m_evt->getPrimaryData() ;
    m_primary_max = m_primary->getShape(0) ;

    m_primary_id = 0 ;  
    m_primary->zero();

    LOG(info) << "CRecorder::setupPrimaryRecording"
              << " primary_max " << m_primary_max 
              ; 
}


void CRecorder::RecordPrimaryVertex(G4PrimaryVertex* vertex)
{
    if(m_primary == NULL || m_primary_id >= m_primary_max ) return ; 

    G4ThreeVector pos = vertex->GetPosition() ;
    G4double time = vertex->GetT0() ;

    G4PrimaryParticle* particle = vertex->GetPrimary();     

    const G4ThreeVector& dir = particle->GetMomentumDirection()  ; 
    G4ThreeVector pol = particle->GetPolarization() ;
  
    G4double energy = particle->GetTotalEnergy()  ; 
    G4double wavelength = h_Planck*c_light/energy ;
    G4double weight = particle->GetWeight() ; 

    m_primary->setQuad(m_primary_id, 0, 0, pos.x()/mm, pos.y()/mm, pos.z()/mm, time/ns  );
    m_primary->setQuad(m_primary_id, 1, 0, dir.x(), dir.y(), dir.z(), weight  );
    m_primary->setQuad(m_primary_id, 2, 0, pol.x(), pol.y(), pol.z(), wavelength/nm  );

    m_primary->setUInt(m_primary_id, 3, 0, 0, 0u );
    m_primary->setUInt(m_primary_id, 3, 0, 1, 0u );
    m_primary->setUInt(m_primary_id, 3, 0, 2, 0u );
    m_primary->setUInt(m_primary_id, 3, 0, 3, 0u );

    m_primary_id += 1 ; 
}

*/



void CRecorder::Summary(const char* msg)
{
    LOG(info) <<  msg
              << " event_id " << m_event_id 
              << " photon_id " << m_photon_id 
              << " record_id " << m_record_id 
              << " step_id " << m_step_id 
              << " m_slot " << m_slot 
              ;
}

