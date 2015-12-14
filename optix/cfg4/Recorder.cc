#include "Recorder.hh"
#include "Format.hh"
#include "OpStatus.hh"

#include "G4RunManager.hh"
#include "G4Event.hh"

#include "G4Track.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4StepStatus.hh"
#include "G4ThreeVector.hh"

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"


#include "NPY.hpp"
#include "NLog.hpp"


void Recorder::init()
{
    if(     strcmp(m_typ,"torch")==0)         m_gen = TORCH ;
    else if(strcmp(m_typ,"cerenkov")==0)      m_gen = CERENKOV ;
    else if(strcmp(m_typ,"scintillation")==0) m_gen = SCINTILLATION ;
    else
         assert(0);

    m_history = NPY<unsigned long long>::make( m_record_max, 1, 2) ; 
    m_history->zero();

    m_photons = NPY<float>::make( m_record_max, 4, 4) ; 
    m_photons->zero();

    m_records = NPY<short>::make( m_record_max, m_steps_per_photon, 2, 4) ; 
    m_records->zero();

    m_fdom = NPY<float>::make(3,1,4);
    m_fdom->zero();

    m_idom = NPY<int>::make(1,1,4);
    m_idom->zero();

}

void Recorder::save()
{
    m_fdom->setQuad(0, 0, m_center_extent ); 
    m_fdom->setQuad(1, 0, m_time_domain ); 

    glm::ivec4 ci ;
    ci.x = 0 ; //m_bounce_max
    ci.y = 0 ; //m_rng_max    
    ci.z = 0 ;   
    ci.w = m_steps_per_photon ; 

    m_idom->setQuad(0, 0, ci );


    m_photons->save("ox%s", m_typ, m_tag, m_det);
    m_records->save("rx%s", m_typ, m_tag, m_det);
    m_history->save("ph%s", m_typ, m_tag, m_det);
    m_fdom->save("fdom%s", m_typ,  m_tag, m_det);
    m_idom->save("idom%s", m_typ,  m_tag, m_det);
}




#define RSAVE(flag, material, slot)  \
{    \
    unsigned int shift = slot*4 ; \
    unsigned long long his = ffs(flag) & 0xF ; \
    unsigned long long mat = material < 0xF ? material : 0xF ; \
    m_seqhis |= his << shift ; \
    m_seqmat |= mat << shift ; \
}   \

unsigned int Recorder::getPointFlag(G4StepPoint* point)
{
    G4StepStatus status = point->GetStepStatus()  ;

    const G4VProcess* process = point->GetProcessDefinedStep() ;
    const G4String& processName = process ? process->GetProcessName() : "NoProc" ; 

    bool transportation = strcmp(processName,"Transportation")==0 ;

    unsigned int flag(0);
    if(strcmp(processName,"OpAbsorption")==0 && status == fPostStepDoItProc )
    {
        flag = BULK_ABSORB ;
    }
    else if(transportation && status == fWorldBoundary )
    {
        flag = SURFACE_ABSORB ;   // kludge
    }
    else if(transportation && status == fGeomBoundary )
    {
        flag = OpBoundaryFlag(m_boundary_status) ; 
    } 
    return flag ; 
}


void Recorder::RecordStep(const G4Step* step)
{
    unsigned int eid = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();
    bool startEvent = eid != m_event_id ; 
    setEventId(eid);

    G4Track* track = step->GetTrack();
    G4int tid = track->GetTrackID();

    G4int sid = track->GetCurrentStepNumber() ;
    assert(tid >= 1 && sid >= 1);   // one-based

    bool startPhoton = tid-1 != m_photon_id ; 
    setPhotonId(tid-1);   // zero-based
    setStepId(sid-1);

    if(startPhoton)
    {
        m_seqhis = 0 ; 
        m_seqmat = 0 ; 
    }

    unsigned int record_id = getRecordId();
    if(record_id >= m_record_max) return ; 

    G4StepPoint* pre  = step->GetPreStepPoint() ; 
    G4StepPoint* post = step->GetPostStepPoint() ; 

    // TODO: cache the relevant process objects, so can just compare pointers

    //  RecordStep is called for all G4Step
    //  each is comprised of pre and post points, so 
    //  see the same points twice : thus need to pick one to record
    //  except for the last 

    unsigned int preFlag  = m_step_id == 0 ? m_gen : getPointFlag(pre);
    unsigned int postFlag = getPointFlag(post) ;
    bool last = (postFlag & (BULK_ABSORB | SURFACE_ABSORB)) != 0 ;


    RecordStepPoint(pre, record_id, m_step_id, preFlag, false );
    if(last)
    {
        RecordStepPoint(post, record_id, m_step_id+1, postFlag, true );
    }


    //if(0)
    if(m_photon_id < 10) 
    {
        G4StepStatus preStatus = pre->GetStepStatus()  ;
        G4StepStatus postStatus = post->GetStepStatus()  ;

        const G4VProcess* preProcess = pre->GetProcessDefinedStep() ;
        const G4String& preProcessName = preProcess ? preProcess->GetProcessName() : "NoPreProc" ; 

        const G4VProcess* postProcess = post->GetProcessDefinedStep() ;
        const G4String& postProcessName = postProcess ? postProcess->GetProcessName() : "NoPostProc" ; 

        if(startPhoton)
        LOG(info) 
              << "\n\n"
              << "Recorder::RecordStep" 
              << " photon_id " << m_photon_id
              ;

        LOG(info) 
              << " [" << m_step_id << "]"
              << " " << OpFlagString(preFlag) << "/" << OpFlagString(postFlag) 
              << " " << OpStepString(preStatus) << "/" << OpStepString(postStatus)
              << " " << OpBoundaryString(m_boundary_status) 
              << " " << preProcessName << "/" << postProcessName 
              << " eid " << m_event_id 
              << " pid " << m_photon_id 
              << " rid " << record_id 
              << "\n"
              << Format(step) ; 
              ;
    }

}



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


struct short4 
{  
   short x ; 
   short y ; 
   short z ; 
   short w ; 
};

struct ushort4 
{  
   unsigned short x ; 
   unsigned short y ; 
   unsigned short z ; 
   unsigned short w ; 
};

union hquad
{   
   short4   short_ ;
   ushort4  ushort_ ;
};  

struct char4
{
   char x ; 
   char y ; 
   char z ; 
   char w ; 
};

struct uchar4
{
   unsigned char x ; 
   unsigned char y ; 
   unsigned char z ; 
   unsigned char w ; 
};

union qquad
{   
   char4   char_   ;
   uchar4  uchar_  ;
};  



void Recorder::RecordStepPoint(const G4StepPoint* point, unsigned int record_id, unsigned int slot, unsigned int flag, bool last)
{
    unsigned int slot_offset =  slot < m_steps_per_photon  ? slot : m_steps_per_photon - 1 ;

    const G4ThreeVector& pos = point->GetPosition();
    const G4ThreeVector& dir = point->GetMomentumDirection();
    const G4ThreeVector& pol = point->GetPolarization();

    G4double time = point->GetGlobalTime();
    G4double energy = point->GetKineticEnergy();
    G4double wavelength = h_Planck*c_light/energy ;
    G4double weight = 1.0 ; 

    unsigned int material = 0 ; 
    RSAVE(flag,material,slot_offset)


    short posx = shortnorm(pos.x()/mm, m_center_extent.x, m_center_extent.w ); 
    short posy = shortnorm(pos.y()/mm, m_center_extent.y, m_center_extent.w ); 
    short posz = shortnorm(pos.z()/mm, m_center_extent.z, m_center_extent.w ); 
    short time_ = shortnorm(time/ns,    m_time_domain.x, m_time_domain.y );

    m_records->setQuad(record_id, slot_offset, 0, posx, posy, posz, time_ );

    unsigned char polx = uchar_( pol.x() );
    unsigned char poly = uchar_( pol.y() );
    unsigned char polz = uchar_( pol.z() );
    unsigned char wavl = uchar_( 255.f*(wavelength/nm - m_boundary_domain.x)/m_boundary_domain.w );

    qquad qaux ; 
    qaux.uchar_.x = 0 ; // TODO:m1 
    qaux.uchar_.y = 0 ; // TODO:m2 
    qaux.char_.z  = 0 ; // TODO:boundary (G4 equivalent ?)
    qaux.uchar_.w = ffs(flag) ; 

    hquad polw ; 
    polw.ushort_.x = polx | poly << 8 ; 
    polw.ushort_.y = polz | wavl << 8 ; 
    polw.ushort_.z = qaux.uchar_.x | qaux.uchar_.y << 8  ;
    polw.ushort_.w = qaux.uchar_.z | qaux.uchar_.w << 8  ;

    m_records->setQuad(record_id, slot_offset, 1, polw.short_.x, polw.short_.y, polw.short_.z, polw.short_.w );  

    if(last)
    {
        m_photons->setQuad(record_id, 0, 0, pos.x()/mm, pos.y()/mm, pos.z()/mm, time/ns  );
        m_photons->setQuad(record_id, 1, 0, dir.x(), dir.y(), dir.z(), weight  );
        m_photons->setQuad(record_id, 2, 0, pol.x(), pol.y(), pol.z(), wavelength/nm  );
        m_photons->setQuad(record_id, 3, 0, 0,0,0,0 );     // TODO: these flags

        LOG(info) << "Recorder::RecordStepPoint"
                  << " record_id " << record_id 
                  << " slot_offset " << slot_offset
                  << " seqhis " << std::hex << m_seqhis << std::dec 
                  << " " << OpFlagSequenceString(m_seqhis) ;

        unsigned long long* history = m_history->getValues() + 2*record_id ;
        *(history+0) = m_seqhis ; 
        *(history+1) = m_seqmat ; 
    }
}

