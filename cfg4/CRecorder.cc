#include <sstream>
#include "CFG4_BODY.hh"
#include "CBoundaryProcess.hh"

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
#include "CGeometry.hh"
#include "CMaterialBridge.hh"
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

/**
CRecorder
==========

Canonical instance is ctor resident of CG4 

**/

CRecorder::CRecorder(Opticks* ok, CGeometry* geometry, bool dynamic) 
   :
   m_ok(ok),
   m_evt(NULL),
   m_geometry(geometry),
   m_material_bridge(NULL),
   m_dynamic(dynamic),
   m_gen(0),

   m_record_max(0),
   m_bounce_max(0),
   m_steps_per_photon(0), 

   m_photons_per_g4event(0),

   m_verbosity(m_ok->hasOpt("steppingdbg") ? 10 : 0),
   m_debug(m_verbosity > 0),
   m_event_id(UINT_MAX),
   m_photon_id(UINT_MAX),
   m_photon_id_prior(UINT_MAX),
   m_step_id(UINT_MAX),
   m_record_id(UINT_MAX),
   m_primary_id(UINT_MAX),

   m_boundary_status(Undefined),
   m_prior_boundary_status(Undefined),

   m_premat(0),
   m_prior_premat(0),

   m_postmat(0),
   m_prior_postmat(0),

   m_seqhis(0),
   m_seqmat(0),
   m_mskhis(0),

   m_seqhis_select(0),
   m_seqmat_select(0),
   m_slot(0),
   m_truncate(false),
   m_step(NULL),

   m_primary(0),
   m_photons(0),
   m_records(0),
   m_history(0),


   m_dynamic_records(NULL),
   m_dynamic_photons(NULL),
   m_dynamic_history(NULL)
{
   
}


void CRecorder::postinitialize()
{
    m_material_bridge = m_geometry->getMaterialBridge();
    assert(m_material_bridge);
}

void CRecorder::setDebug(bool debug)
{
    m_debug = debug ; 
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




unsigned CRecorder::getRecordMax()
{
    return m_record_max ; 
}


int CRecorder::getEventId()
{
   return m_event_id ; 
}
int CRecorder::getPhotonId()
{
   return m_photon_id ; 
}
int CRecorder::getPhotonIdPrior()
{
   return m_photon_id_prior ; 
}


int CRecorder::getParentId()
{
   return m_parent_id ; 
}
int CRecorder::getStepId()
{
   return m_step_id ; 
}
int CRecorder::getRecordId()
{
   return m_record_id ; 
}





void CRecorder::setEventId(int event_id)
{
    m_event_id = event_id ; 
}
void CRecorder::setPhotonId(int photon_id)
{
    m_photon_id_prior = m_photon_id ; 
    m_photon_id = photon_id ; 
}

void CRecorder::setParentId(int parent_id)
{
    m_parent_id = parent_id ; 
}
void CRecorder::setPrimaryId(int primary_id)
{
    m_primary_id = primary_id ; 
}


void CRecorder::setStage(CStage::CStage_t stage)
{
    m_stage = stage ; 
}


std::string CRecorder::description()
{
    std::stringstream ss ; 
    ss << std::setw(10) << CStage::Label(m_stage)
       << " evt " << std::setw(7) << m_event_id
       << " pho " << std::setw(7) << m_photon_id 
       << " par " << std::setw(7) << m_parent_id
       << " pri " << std::setw(7) << m_primary_id
       << " ste " << std::setw(4) << m_step_id 
       << " rid " << std::setw(4) << m_record_id 
       << " slt " << std::setw(4) << m_slot
       << " pre " << std::setw(7) << getPreGlobalTime(m_step)/ns
       << " pst " << std::setw(7) << getPostGlobalTime(m_step)/ns
       << ( m_dynamic ? " DYNAMIC " : " STATIC " )
       ;

   return ss.str();
}








void CRecorder::setStepId(int step_id)
{
    m_step_id = step_id ; 
}
int CRecorder::defineRecordId()   
{
   return m_photons_per_g4event*m_event_id + m_photon_id ; 
}

void CRecorder::setRecordId(int record_id)
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


void CRecorder::setEvent(OpticksEvent* evt)
{
    m_evt = evt ; 
    assert(m_evt && m_evt->isG4());
}

void CRecorder::initEvent(OpticksEvent* evt)
{
    setEvent(evt);

    m_c4.u = 0u ; 

    m_photons_per_g4event = m_evt->getNumPhotonsPerG4Event() ; 
    m_record_max = m_evt->getNumPhotons();   // from the genstep summation

    m_bounce_max = m_evt->getBounceMax();
    m_steps_per_photon = m_evt->getMaxRec() ;    

    LOG(info) << "CRecorder::initEvent"
              << " dynamic " << ( m_dynamic ? "DYNAMIC(CPU style)" : "STATIC(GPU style)" )
              << " record_max " << m_record_max
              << " bounce_max  " << m_bounce_max 
              << " steps_per_photon " << m_steps_per_photon 
              << " photons_per_g4event " << m_photons_per_g4event
              << " num_g4event " << m_evt->getNumG4Event() 
              << " isStep " << m_step  
              ;

    if(m_dynamic)
    {
        assert(m_record_max == 0 );

        // shapes must match OpticksEvent::createBuffers
        // TODO: avoid this duplicity 

        m_dynamic_records = NPY<short>::make(1, m_steps_per_photon, 2, 4) ;
        m_dynamic_records->zero();

        m_dynamic_photons = NPY<float>::make(1, 4, 4) ;
        m_dynamic_photons->zero();

        m_dynamic_history = NPY<unsigned long long>::make(1, 1, 2) ;
        m_dynamic_history->zero();

    } 
    else
    {
        assert(m_record_max > 0 );
    }

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

void CRecorder::decrementSlot()
{
    //if(m_slot == 0 || m_truncate)
    if(m_slot == 0 )
    {
        LOG(warning) << "CRecorder::decrementSlot SKIPPING"
                     << " slot " << m_slot 
                     << " truncate " << m_truncate 
                      ;
        return ;
    }
    m_slot -= 1; 
}

unsigned CRecorder::getSlot()
{
    return m_slot ; 
}

void CRecorder::setSlot(unsigned slot)
{
   // needed for reemission continuation
    m_slot = slot ; 
}

void CRecorder::startPhoton()
{
    //LOG(trace) << "CRecorder::startPhoton" ; 

    //if(m_record_id % 10000 == 0) Summary("CRecorder::startPhoton(%10k)") ;

    //assert(m_step_id == 0);  <-- when not always true when skipping same material steps 

    m_c4.u = 0u ; 

    m_boundary_status = Undefined ; 
    m_prior_boundary_status = Undefined ; 

    m_premat = 0 ; 
    m_prior_premat = 0 ; 

    m_postmat = 0 ; 
    m_prior_postmat = 0 ; 

    m_seqmat = 0 ; 
    m_seqhis = 0 ; 
    m_mskhis = 0 ; 

    //m_seqhis_select = 0xfbbbbbbbcd ;
    //m_seqhis_select = 0x8cbbbbbc0 ;
    m_seqhis_select = 0x8bd ;

    m_slot = 0 ; 
    m_truncate = false ; 

    if(m_debug) Clear();
}









#ifdef USE_CUSTOM_BOUNDARY
void CRecorder::setBoundaryStatus(DsG4OpBoundaryProcessStatus boundary_status, unsigned int premat, unsigned int postmat)
#else
void CRecorder::setBoundaryStatus(G4OpBoundaryProcessStatus boundary_status, unsigned int premat, unsigned int postmat)
#endif
{
    // this is invoked before RecordStep is called from SteppingAction
    m_prior_boundary_status = m_boundary_status ; 
    m_prior_premat = m_premat ; 
    m_prior_postmat = m_postmat ; 

    m_boundary_status = boundary_status ; 
    m_premat = premat ; 
    m_postmat = postmat ;
}
double CRecorder::getPreGlobalTime(const G4Step* step)
{
    const G4StepPoint* point  = step->GetPreStepPoint() ; 
    return point->GetGlobalTime();
}
double CRecorder::getPostGlobalTime(const G4Step* step)
{
    const G4StepPoint* point  = step->GetPostStepPoint() ; 
    return point->GetGlobalTime();
}

 
void CRecorder::setStep(const G4Step* step)
{
    m_step = step ; 
}
bool CRecorder::RecordStep()
{
    assert(m_step && "MUST setStep FIRST");

    const G4StepPoint* pre  = m_step->GetPreStepPoint() ; 
    const G4StepPoint* post = m_step->GetPostStepPoint() ; 

    // shunt flags by 1 relative to steps, in order to set the generation code on first step
    // this doesnt miss flags, as record both pre and post at last step    

    unsigned preFlag = m_slot == 0 ? m_gen : OpPointFlag(pre,  m_prior_boundary_status, m_stage);
    unsigned postFlag =                      OpPointFlag(post, m_boundary_status      , m_stage);

    bool lastPost = (postFlag & (BULK_ABSORB | SURFACE_ABSORB | SURFACE_DETECT)) != 0 ;

    bool surfaceAbsorb = (postFlag & (SURFACE_ABSORB | SURFACE_DETECT)) != 0 ;

    bool preSkip = m_prior_boundary_status == StepTooSmall ;  

    bool matSwap = m_boundary_status == StepTooSmall ; 

    unsigned preMat  = matSwap ? m_postmat : m_premat ;

    unsigned postMat = ( matSwap || m_postmat == 0 )  ? m_premat  : m_postmat ;

    if(surfaceAbsorb) postMat = m_postmat ; 


    bool done = false ; 

    // skip the pre, but the post becomes the pre at next step where will be taken 
    // 1-based material indices, so zero can represent None
    //
    //   RecordStepPoint records into m_slot (if < m_steps_per_photon) and increments m_slot
    // 

    if(!preSkip)
    {
       done = RecordStepPoint( pre, preFlag, preMat, m_prior_boundary_status, PRE ); 
    }

    if(lastPost && !done)
    {
       done = RecordStepPoint( post, postFlag, postMat, m_boundary_status, POST ); 
    }

    // when not *lastPost* the post step will become the pre step at next RecordStep
    // so every step point is recorded 
    //
    // in case of reemission a BULK_ABSORB is reincarnated with a BULK_REEMIT that 
    // needs to replace the BULK_ABSORB that was layed down in an earlier lastPost
    //
    // It looks like static mode will mostly succeed to scrub the AB and replace with RE 
    // just by decrementing m_slot and running again
    // but dynamic mode will have an extra record.
    //
    //  Decrementing m_slot and running again is effectively replacing 
    //  the final "AB" (post) RecordStepPoint 
    //  with the subsequent "RE" (pre) RecordStepPoint  
    //

    return done ;
}


#ifdef USE_CUSTOM_BOUNDARY
bool CRecorder::RecordStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, DsG4OpBoundaryProcessStatus boundary_status, const char* label)
#else
bool CRecorder::RecordStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, G4OpBoundaryProcessStatus boundary_status, const char* label)
#endif
{
    bool absorb = ( flag & (BULK_ABSORB | SURFACE_ABSORB | SURFACE_DETECT)) != 0 ;

    unsigned int slot =  m_slot < m_steps_per_photon  ? m_slot : m_steps_per_photon - 1 ;
    // constrain slot to recording inclusive range (0,m_steps_per_photon-1) 

    m_truncate = slot == m_steps_per_photon - 1 ; 

    if(flag == 0)
    {
       if(!(boundary_status == SameMaterial || boundary_status == Undefined))
            LOG(warning) << " boundary_status not handled : " << OpBoundaryAbbrevString(boundary_status) ; 
    }

    //Dump(label,  slot, point, boundary_status );

    unsigned long long shift = slot*4ull ;     // 4-bits of shift for each slot 
    unsigned long long msk = 0xFull << shift ; 
    unsigned long long his = BBit::ffs(flag) & 0xFull ; 
    unsigned long long mat = material < 0xFull ? material : 0xFull ; 

    m_seqhis =  (m_seqhis & (~msk)) | (his << shift) ; 
    m_seqmat =  (m_seqmat & (~msk)) | (mat << shift) ; 

    m_mskhis |= flag ;   // <-- hmm decrementing m_slot and running again will not scrub the AB from the mask

    RecordStepPoint(slot, point, flag, material, label);

    double time = point->GetGlobalTime();
    if(m_debug) Collect(point, flag, material, boundary_status, m_seqhis, m_seqmat, time);

    m_slot += 1 ;   

    // m_slot is incremented regardless of truncation, 
    // only local *slot* is constrained to recording range

    bool truncate = m_slot > m_bounce_max  ;  
    bool done = truncate || absorb ;   

    if(done && m_dynamic)
    {
        m_records->add(m_dynamic_records);
    }


    if(m_debug)
       LOG(info) << "CRecorder::RecordStepPoint"
                 << " m_slot " << m_slot 
                 << " slot " << slot 
                 << " flag " << std::hex << BBit::ffs(flag) << std::dec
                 << " done " << ( done ? "Y" : "N" )
                 << " truncate " << ( truncate ? "Y" : "N" )
                 << description()
                 ; 

    return done ; 

  /*
      edge case of reemission when already reached truncation means that the decrementSlot 
      can change truncation state ...  
  */
}


#ifdef USE_CUSTOM_BOUNDARY
void CRecorder::Collect(const G4StepPoint* point, unsigned int flag, unsigned int material, DsG4OpBoundaryProcessStatus boundary_status, unsigned long long seqhis, unsigned long long seqmat, double time)
#else
void CRecorder::Collect(const G4StepPoint* point, unsigned int flag, unsigned int material, G4OpBoundaryProcessStatus boundary_status, unsigned long long seqhis, unsigned long long seqmat, double time)
#endif
{
    assert(m_debug);
    m_points.push_back(new G4StepPoint(*point));
    m_flags.push_back(flag);
    m_materials.push_back(material);
    m_bndstats.push_back(boundary_status);  // will duplicate the status for the last step
    m_seqhis_dbg.push_back(seqhis);
    m_seqmat_dbg.push_back(seqmat);
    m_times.push_back(time);
}




#ifdef USE_CUSTOM_BOUNDARY
DsG4OpBoundaryProcessStatus CRecorder::getBoundaryStatus()
#else
G4OpBoundaryProcessStatus CRecorder::getBoundaryStatus()
#endif
{
   return m_boundary_status ; 
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

    // looks like static mode will succeed to scrub the AB and replace with RE 
    // just by decrementing m_slot and running again
    // but dynamic mode will have an extra record
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
    target->setUInt(target_record_id, 3, 0, 3, m_mskhis );

    // in static case directly populate the pre-sized photon buffer
    // in dynamic case populate the single photon buffer first and then 
    // add that to the photons below

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

    NPY<unsigned long long>* h_target = m_dynamic ? m_dynamic_history : m_history ; 

    unsigned long long* history = h_target->getValues() + 2*target_record_id ;
    *(history+0) = m_seqhis ; 
    *(history+1) = m_seqmat ; 

    if(m_dynamic)
    {
        m_history->add(m_dynamic_history);
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

#ifdef USE_CUSTOM_BOUNDARY
void CRecorder::Dump(const G4ThreeVector& origin, unsigned int index, const G4StepPoint* point, DsG4OpBoundaryProcessStatus boundary_status, const char* matname )
#else
void CRecorder::Dump(const G4ThreeVector& origin, unsigned int index, const G4StepPoint* point, G4OpBoundaryProcessStatus boundary_status, const char* matname )
#endif
{
    std::string bs = OpBoundaryAbbrevString(boundary_status) ;
    std::cout << std::setw(7) << index << " " << std::setw(18) << matname << " " << Format(point, origin, bs.c_str()) << std::endl ;
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
              << " " << m_material_bridge->MaterialSequence(m_seqmat) 
              ;

    if(!m_debug) return ; 

    G4ThreeVector origin ;
    if(m_points.size() > 0) origin = m_points[0]->GetPosition();

    for(unsigned int i=0 ; i<m_points.size() ; i++) 
    {
#ifdef USE_CUSTOM_BOUNDARY
       DsG4OpBoundaryProcessStatus bst = m_bndstats[i] ;
#else
       G4OpBoundaryProcessStatus bst = m_bndstats[i] ;
#endif
       unsigned mat = m_materials[i] ;
       const char* matname = ( mat == 0 ? "-" : m_material_bridge->getMaterialName(mat-1)  ) ;

       Dump(origin, i, m_points[i], bst, matname );

       //std::cout << std::hex << m_seqhis_dbg[i] << std::dec << std::endl ; 
       //std::cout << std::hex << m_seqmat_dbg[i] << std::dec << std::endl ; 
    }
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
    m_times.clear();
}

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

