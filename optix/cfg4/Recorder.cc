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

#include "Recorder.icc"



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

unsigned int Recorder::getPointFlag(const G4StepPoint* point)
{
    G4StepStatus status = point->GetStepStatus()  ;

    // TODO: cache the relevant process objects, so can just compare pointers ?
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
    // seeing duplicate StepPoints differ-ing only in the volume
    // skip these by early exit
    // TODO:  more careful handling for correct material recording
    //


    unsigned int eid = G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID();

    G4Track* track = step->GetTrack();
    G4int photon_id = track->GetTrackID() - 1;
    G4int step_id  = track->GetCurrentStepNumber() - 1 ;

    setEventId(eid);
    setPhotonId(photon_id);   
    setStepId(step_id);

    unsigned int record_id = m_photons_per_event*m_event_id + m_photon_id ; 
    setRecordId(record_id);

    if(record_id >= m_record_max) return ; 

    bool first = step_id == 0; 
    if(first)
    {
        m_seqhis = 0 ; 
        m_seqmat = 0 ; 
        m_slot = 0 ; 
        Clear();
    }

    const G4StepPoint* pre  = step->GetPreStepPoint() ; 
    const G4StepPoint* post = step->GetPostStepPoint() ; 

    unsigned int preFlag  = m_step_id == 0 ? m_gen : getPointFlag(pre);
    unsigned int postFlag = getPointFlag(post) ;

    if(m_boundary_status == StepTooSmall ) return ;  

    bool last = (postFlag & (BULK_ABSORB | SURFACE_ABSORB)) != 0 ;

    RecordStepPoint(pre, preFlag, false );
    if(last)
        RecordStepPoint(post, postFlag, true );

    if(last)
    {
        assert(m_flags.size() == m_points.size());
        bool issue = false ; 
        for(unsigned int i=0 ; i < m_flags.size() ; i++) if(m_flags[i] == 0 || m_flags[i] == NAN_ABORT) issue = true ; 
        if(m_record_id < 10 || issue) Dump("Recorder::RecordStep") ;
    }
}

void Recorder::Dump(const char* msg)
{
    LOG(info) << msg 
              << " seqhis " << std::hex << m_seqhis << std::dec 
              << " " << OpFlagSequenceString(m_seqhis) ;

    for(unsigned int i=0 ; i<m_points.size() ; i++) 
    {
       std::cout << Format(m_points[i]) << std::endl ;
       G4OpBoundaryProcessStatus bst = m_bndstats[i] ;
       if( i < m_points.size() - 1 && bst != FresnelRefraction) std::cout << OpBoundaryString(bst) << std::endl ; 
    }
}

void Recorder::Collect(const G4StepPoint* point, unsigned int flag, G4OpBoundaryProcessStatus boundary_status)
{
    m_points.push_back(new G4StepPoint(*point));
    m_flags.push_back(flag);
    m_bndstats.push_back(boundary_status);  // will duplicate the status for the last step
}

void Recorder::Clear()
{
    for(unsigned int i=0 ; i < m_points.size() ; i++) delete m_points[i] ;
    m_points.clear();
    m_flags.clear();
    m_bndstats.clear();
}

void Recorder::RecordStepPoint(const G4StepPoint* point, unsigned int flag, bool last)
{
    Collect(point, flag, m_boundary_status);

    unsigned int slot =  m_slot < m_steps_per_photon  ? m_slot : m_steps_per_photon - 1 ;
    m_slot += 1 ; 

    const G4ThreeVector& pos = point->GetPosition();
    const G4ThreeVector& dir = point->GetMomentumDirection();
    const G4ThreeVector& pol = point->GetPolarization();

    G4double time = point->GetGlobalTime();
    G4double energy = point->GetKineticEnergy();
    G4double wavelength = h_Planck*c_light/energy ;
    G4double weight = 1.0 ; 

    unsigned int material = 0 ; 

    RSAVE(flag,material,slot )

    short posx = shortnorm(pos.x()/mm, m_center_extent.x, m_center_extent.w ); 
    short posy = shortnorm(pos.y()/mm, m_center_extent.y, m_center_extent.w ); 
    short posz = shortnorm(pos.z()/mm, m_center_extent.z, m_center_extent.w ); 
    short time_ = shortnorm(time/ns,    m_time_domain.x, m_time_domain.y );

    m_records->setQuad(m_record_id, slot, 0, posx, posy, posz, time_ );

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

    m_records->setQuad(m_record_id, slot, 1, polw.short_.x, polw.short_.y, polw.short_.z, polw.short_.w );  

    if(last)
    {
        m_photons->setQuad(m_record_id, 0, 0, pos.x()/mm, pos.y()/mm, pos.z()/mm, time/ns  );
        m_photons->setQuad(m_record_id, 1, 0, dir.x(), dir.y(), dir.z(), weight  );
        m_photons->setQuad(m_record_id, 2, 0, pol.x(), pol.y(), pol.z(), wavelength/nm  );
        m_photons->setQuad(m_record_id, 3, 0, 0,0,0,0 );     // TODO: these flags

        unsigned long long* history = m_history->getValues() + 2*m_record_id ;
        *(history+0) = m_seqhis ; 
        *(history+1) = m_seqmat ; 
    }


}

