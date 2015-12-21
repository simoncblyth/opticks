#include "Recorder.hh"
#include "Format.hh"

#include "Opticks.hh"
#include "OpStatus.hh"

#include "G4Track.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4StepStatus.hh"
#include "G4ThreeVector.hh"

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "NumpyEvt.hpp"
#include "NPY.hpp"
#include "NLog.hpp"

#include "Recorder.icc"

void Recorder::init()
{
    m_record_max = m_evt->getNumPhotons(); 
    m_steps_per_photon = m_evt->getMaxRec() ;    

    LOG(info) << "Recorder::init"
              << " record_max " << m_record_max
              << " steps_per_photon " << m_steps_per_photon 
              ;

    m_evt->zero();

    m_history = m_evt->getSequenceData();
    m_photons = m_evt->getPhotonData();
    m_records = m_evt->getRecordData();

    const char* typ = m_evt->getTyp();
    assert(strcmp(typ,Opticks::torch_) == 0);
    m_gen = Opticks::SourceCode(typ);
    assert( m_gen == TORCH );
}


unsigned int Recorder::getPointFlag(const G4StepPoint* point, const G4OpBoundaryProcessStatus bst)
{
    G4StepStatus status = point->GetStepStatus()  ;
    // TODO: cache the relevant process objects, so can just compare pointers ?
    const G4VProcess* process = point->GetProcessDefinedStep() ;
    const G4String& processName = process ? process->GetProcessName() : "NoProc" ; 

    bool transportation = strcmp(processName,"Transportation") == 0 ;
    bool scatter = strcmp(processName, "OpRayleigh") == 0 ; 
    bool absorption = strcmp(processName, "OpAbsorption") == 0 ;

    unsigned int flag(0);
    if(absorption && status == fPostStepDoItProc )
    {
        flag = BULK_ABSORB ;
    }
    else if(scatter && status == fPostStepDoItProc )
    {
        flag = BULK_SCATTER ;
    }
    else if(transportation && status == fWorldBoundary )
    {
        flag = SURFACE_ABSORB ;   // kludge for fWorldBoundary - no surface handling yet 
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

    const G4StepPoint* pre  = step->GetPreStepPoint() ; 
    const G4StepPoint* post = step->GetPostStepPoint() ; 

    // TODO:  material recording
    /*
    G4VPhysicalVolume* prePV  = pre->GetPhysicalVolume();
    G4VPhysicalVolume* postPV  = post->GetPhysicalVolume();
    */

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


    // StepTooSmall occurs at boundaries with pre/post StepPoints 
    // almost the same differing only in their associated volume

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
              << " " << Opticks::FlagSequence(m_seqhis) ;

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

    const glm::vec4& sd = m_evt->getSpaceDomain() ; 
    const glm::vec4& td = m_evt->getTimeDomain() ; 
    const glm::vec4& wd = m_evt->getWavelengthDomain() ; 

    short posx = shortnorm(pos.x()/mm, sd.x, sd.w ); 
    short posy = shortnorm(pos.y()/mm, sd.y, sd.w ); 
    short posz = shortnorm(pos.z()/mm, sd.z, sd.w ); 
    short time_ = shortnorm(time/ns,   td.x, td.y );

    m_records->setQuad(m_record_id, slot, 0, posx, posy, posz, time_ );


    unsigned char polx = uchar_( pol.x() );
    unsigned char poly = uchar_( pol.y() );
    unsigned char polz = uchar_( pol.z() );
    unsigned char wavl = uchar_( 255.f*(wavelength/nm - wd.x)/wd.w );

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

    /*
    if(m_record_id < 10000)
        LOG(info) << "Recorder::RecordStepPoint" 
                  << " record_id " << m_record_id
                  << " m_slot " << m_slot 
                  << " slot " << slot 
                  << " time " << time
                  << " time_ " << time_
                  << " ns " << ns
                  << " time_domain.x " << m_time_domain.x
                  << " time_domain.y " << m_time_domain.y
                  ;
     */ 

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

