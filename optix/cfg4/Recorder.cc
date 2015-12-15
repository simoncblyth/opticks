#include "Recorder.hh"
#include "Format.hh"
#include "OpStatus.hh"

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


unsigned int Recorder::getPointFlag(const G4StepPoint* point, const G4OpBoundaryProcessStatus bst)
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
        flag = OpBoundaryFlag(bst) ;
    } 
    return flag ; 
}


void Recorder::setBoundaryStatus(G4OpBoundaryProcessStatus boundary_status)
{
    // this is invoked before RecordStep is called from SteppingAction
    m_prior_boundary_status = m_boundary_status ; 
    m_boundary_status = boundary_status ; 
}

void Recorder::startPhoton()
{
    if(m_record_id % 10000 == 0)
    LOG(info) << "Recorder::startPhoton"
              << " event_id " << m_event_id 
              << " photon_id " << m_photon_id 
              << " record_id " << m_record_id 
              << " step_id " << m_step_id 
              ;

    assert(m_step_id == 0);

    m_prior_boundary_status = Undefined ; 
    m_boundary_status = Undefined ; 

    m_seqhis = 0 ; 
    //m_seqhis_select = 0xfbbbbbbbcd ;
    m_seqhis_select = 0x8cbbbbbc0 ;
    m_seqmat = 0 ; 
    m_slot = 0 ; 
    Clear();
}

void Recorder::RecordStep(const G4Step* step)
{
    // seeing duplicate StepPoints differ-ing only in the volume
    // skip these by early exit
    // TODO:  more careful handling for correct material recording

    const G4StepPoint* pre  = step->GetPreStepPoint() ; 
    const G4StepPoint* post = step->GetPostStepPoint() ; 

    unsigned int preFlag(0);
    unsigned int postFlag(0);

    if(m_step_id == 0)
    {
        preFlag = m_gen ;         
        postFlag = getPointFlag(post, m_boundary_status) ;
    }
    else
    {
        preFlag  = getPointFlag(pre, m_prior_boundary_status);
        postFlag = getPointFlag(post, m_boundary_status) ;
    }

    bool last = (postFlag & (BULK_ABSORB | SURFACE_ABSORB)) != 0 ;


    if( m_prior_boundary_status != StepTooSmall)
    RecordStepPoint(pre, preFlag, m_prior_boundary_status, false );

    if(last)
        RecordStepPoint(post, postFlag, m_boundary_status, true );
    

    if(last)
    {
        bool issue = hasIssue();
        if(m_record_id < 10 || issue || m_seqhis == m_seqhis_select ) Dump("Recorder::RecordStep") ;
    }
}

bool Recorder::hasIssue()
{
    unsigned int npoints = m_points.size() ;
    assert(m_flags.size() == npoints);
    assert(m_bndstats.size() == npoints);

    bool issue = false ; 
    for(unsigned int i=0 ; i < npoints ; i++) 
    {
       if(m_flags[i] == 0 || m_flags[i] == NAN_ABORT) issue = true ; 
    }
    return issue ; 
}


void Recorder::Dump(const char* msg)
{
    LOG(info) << msg 
              << " record_id " << std::setw(7) << m_record_id
              << " seqhis " << std::hex << m_seqhis << std::dec 
              << " " << OpFlagSequenceString(m_seqhis) ;

    for(unsigned int i=0 ; i<m_points.size() ; i++) 
    {
       //unsigned long long seqhis = m_seqhis_dbg[i] ;
       G4OpBoundaryProcessStatus bst = m_bndstats[i] ;
       std::string bs = OpBoundaryAbbrevString(bst) ;
       //std::cout << std::hex << seqhis << std::dec << std::endl ; 
       std::cout << std::setw(7) << i << " " << Format(m_points[i], bs.c_str()) << std::endl ;
    }
}

void Recorder::Collect(const G4StepPoint* point, unsigned int flag, G4OpBoundaryProcessStatus boundary_status, unsigned long long seqhis)
{
    m_points.push_back(new G4StepPoint(*point));
    m_flags.push_back(flag);
    m_bndstats.push_back(boundary_status);  // will duplicate the status for the last step
    //m_seqhis_dbg.push_back(seqhis);
}

void Recorder::Clear()
{
    for(unsigned int i=0 ; i < m_points.size() ; i++) delete m_points[i] ;
    m_points.clear();
    m_flags.clear();
    m_bndstats.clear();
    //m_seqhis_dbg.clear();
}


void Recorder::RecordStepPoint(const G4StepPoint* point, unsigned int flag, G4OpBoundaryProcessStatus boundary_status, bool last)
{
    if(flag == 0x1 << 14) LOG(warning) << "Recorder::RecordStepPoint bad flag " << flag ;


    unsigned int slot =  m_slot < m_steps_per_photon  ? m_slot : m_steps_per_photon - 1 ;
    unsigned int material = 0 ; 

    // * masked combination needed (not just m_seqhis |= ) 
    //   in order to correctly handle truncation overwrite
    //
    // * all ingredients must be 64bit otherwise slips down to 32bit 
    //   causing wraparounds, causing bad flags

    unsigned long long shift = slot*4ull ; 
    unsigned long long msk = 0xFull << shift ; 
    unsigned long long his = ffs(flag) & 0xFull ; 
    unsigned long long mat = material < 0xFull ? material : 0xFull ; 
    m_seqhis =  (m_seqhis & (~msk)) | (his << shift) ; 
    m_seqmat =  (m_seqmat & (~msk)) | (mat << shift) ; 

    Collect(point, flag, boundary_status, m_seqhis);

    m_slot += 1 ; 

    const G4ThreeVector& pos = point->GetPosition();
    const G4ThreeVector& dir = point->GetMomentumDirection();
    const G4ThreeVector& pol = point->GetPolarization();

    G4double time = point->GetGlobalTime();
    G4double energy = point->GetKineticEnergy();
    G4double wavelength = h_Planck*c_light/energy ;
    G4double weight = 1.0 ; 

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

