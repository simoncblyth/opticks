#include "Recorder.hh"
#include "Format.hh"

#include "Opticks.hh"
#include "OpStatus.hh"

#include "G4Track.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4StepStatus.hh"
#include "G4ThreeVector.hh"
#include "G4PrimaryVertex.hh"
#include "G4PrimaryParticle.hh"


#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include "NumpyEvt.hpp"
#include "NPY.hpp"
#include "NLog.hpp"

#include "Recorder.icc"


/*
Truncation that matches optixrap-/cu/generate.cu::

    generate...

    int bounce = 0 ;
    int slot = 0 ;
    int slot_min = photon_id*MAXREC ;       // eg 0 for photon_id=0
    int slot_max = slot_min + MAXREC - 1 ;  // eg 9 for photon_id=0, MAXREC=10
    int slot_offset = 0 ;

    while( bounce < bounce_max )
    {
        bounce++

        rtTrace...

        slot_offset =  slot < MAXREC  ? slot_min + slot : slot_max ;

          // eg 0,1,2,3,4,5,6,7,8,9,9,9,9,9,....  if bounce_max were greater than MAXREC
          //    0,1,2,3,4,5,6,7,8,9       for bounce_max = 9, MAXREC = 10 

        RSAVE(..., slot, slot_offset)...
        slot++ ;

        propagate_to_boundary...
        propagate_at_boundary... 
    }

    slot_offset =  slot < MAXREC  ? slot_min + slot : slot_max ;

    RSAVE(..., slot, slot_offset)


Consider truncated case with bounce_max = 9, MAXREC = 10 

* last while loop starts at bounce = 8 
* RSAVE inside the loop invoked with bounce=1:9 
  and then once more beyond the while 
  for a total of 10 RSAVEs 


*/

const char* Recorder::PRE  = "PRE" ; 
const char* Recorder::POST = "POST" ; 


void Recorder::init()
{
    m_record_max = m_evt->getNumPhotons(); 
    m_bounce_max = m_evt->getBounceMax();
    m_steps_per_photon = m_evt->getMaxRec() ;    

    LOG(info) << "Recorder::init"
              << " record_max " << m_record_max
              << " bounce_max  " << m_bounce_max 
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


void Recorder::setupPrimaryRecording()
{
    m_evt->prepareForPrimaryRecording();

    m_primary = m_evt->getPrimaryData() ;
    m_primary_max = m_primary->getShape(0) ;

    m_primary_id = 0 ;  
    m_primary->zero();

    LOG(info) << "Recorder::setupPrimaryRecording"
              << " primary_max " << m_primary_max 
              ; 
 
}

void Recorder::RecordPrimaryVertex(G4PrimaryVertex* vertex)
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

    unsigned int ux = 0u ; 
    unsigned int uy = 0u ; 
    unsigned int uz = 0u ; 
    unsigned int uw = 0u ; 

    m_primary->setUInt(m_primary_id, 3, 0, 0, ux );
    m_primary->setUInt(m_primary_id, 3, 0, 1, uy );
    m_primary->setUInt(m_primary_id, 3, 0, 2, uz );
    m_primary->setUInt(m_primary_id, 3, 0, 3, uw );

    m_primary_id += 1 ; 

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


void Recorder::Summary(const char* msg)
{
    LOG(info) <<  msg
              << " event_id " << m_event_id 
              << " photon_id " << m_photon_id 
              << " record_id " << m_record_id 
              << " step_id " << m_step_id 
              << " m_slot " << m_slot 
              ;
}


void Recorder::startPhoton()
{
    //LOG(info) << "Recorder::startPhoton" ; 

    if(m_record_id % 10000 == 0) Summary("Recorder::startPhoton") ;

    assert(m_step_id == 0);

    m_prior_boundary_status = Undefined ; 
    m_boundary_status = Undefined ; 

    m_seqhis = 0 ; 
    //m_seqhis_select = 0xfbbbbbbbcd ;
    m_seqhis_select = 0x8cbbbbbc0 ;
    m_seqmat = 0 ; 
    m_slot = 0 ; 
    m_truncate = false ; 

    Clear();
}

bool Recorder::RecordStep(const G4Step* step)
{
    const G4StepPoint* pre  = step->GetPreStepPoint() ; 
    const G4StepPoint* post = step->GetPostStepPoint() ; 

    unsigned int preFlag ; 
    unsigned int postFlag ; 

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

    bool postLast = (postFlag & (BULK_ABSORB | SURFACE_ABSORB)) != 0 ;
    bool preSkip = m_prior_boundary_status == StepTooSmall ;
    bool truncate = false ; 

    if(!preSkip)
       truncate = RecordStepPoint( pre, preFlag, m_prior_boundary_status, PRE ); 

    if(postLast && !truncate)
       truncate = RecordStepPoint( post, postFlag, m_boundary_status, POST ); 

    // when not postLast the post step will become the pre step at next RecordStep

    return truncate ;
}

bool Recorder::RecordStepPoint(const G4StepPoint* point, unsigned int flag, G4OpBoundaryProcessStatus boundary_status, const char* label)
{
    bool absorb = ( flag & (BULK_ABSORB | SURFACE_ABSORB)) != 0 ;

    unsigned int slot =  m_slot < m_steps_per_photon  ? m_slot : m_steps_per_photon - 1 ;

    //Dump(label,  slot, point, boundary_status );
 
    RecordStepPoint(slot, point, flag, label);

    Collect(point, flag, boundary_status, m_seqhis);
    m_slot += 1 ; 
    bool truncate = m_slot > m_bounce_max  ;  

    if(truncate || absorb)
    {
        RecordPhoton( point );
        //Dump("Recorder::RecordStepPoint");
        return true ; 
    }
    return false ; 
}


void Recorder::RecordStepPoint(unsigned int slot, const G4StepPoint* point, unsigned int flag, const char* label )
{
    unsigned int material = 0 ; 
    unsigned long long shift = slot*4ull ;   

    unsigned long long msk = 0xFull << shift ; 
    unsigned long long his = ffs(flag) & 0xFull ; 
    unsigned long long mat = material < 0xFull ? material : 0xFull ; 
    m_seqhis =  (m_seqhis & (~msk)) | (his << shift) ; 
    m_seqmat =  (m_seqmat & (~msk)) | (mat << shift) ; 


    /*
    LOG(info) << "Recorder::RecordStepPoint" 
              << " label " << label 
              << " m_record_id " << m_record_id 
              << " m_step_id " << m_step_id 
              << " m_slot " << m_slot 
              << " slot " << slot 
              << " flag " << flag
              << " his " << his
              << " shift " << shift 
              << " m_seqhis " << std::hex << m_seqhis << std::dec 
              ;
    */

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
}



void Recorder::RecordPhoton(const G4StepPoint* point)
{
    const G4ThreeVector& pos = point->GetPosition();
    const G4ThreeVector& dir = point->GetMomentumDirection();
    const G4ThreeVector& pol = point->GetPolarization();

    G4double time = point->GetGlobalTime();
    G4double energy = point->GetKineticEnergy();
    G4double wavelength = h_Planck*c_light/energy ;
    G4double weight = 1.0 ; 

    m_photons->setQuad(m_record_id, 0, 0, pos.x()/mm, pos.y()/mm, pos.z()/mm, time/ns  );
    m_photons->setQuad(m_record_id, 1, 0, dir.x(), dir.y(), dir.z(), weight  );
    m_photons->setQuad(m_record_id, 2, 0, pol.x(), pol.y(), pol.z(), wavelength/nm  );

    unsigned int ux = m_slot ;  // untruncated 
    unsigned int uy = 0u ; 
    unsigned int uz = 0u ; 
    unsigned int uw = 0u ; 

    m_photons->setUInt(m_record_id, 3, 0, 0, ux );
    m_photons->setUInt(m_record_id, 3, 0, 1, uy );
    m_photons->setUInt(m_record_id, 3, 0, 2, uz );
    m_photons->setUInt(m_record_id, 3, 0, 3, uw );


    // generate.cu
    //
    //  (x)  p.flags.i.x = prd.boundary ;   // last boundary
    //  (y)  p.flags.u.y = s.identity.w ;   // sensorIndex  >0 only for cathode hits
    //  (z)  p.flags.u.z = s.index.x ;      // material1 index  : redundant with boundary  
    //  (w)  p.flags.u.w |= s.flag ;        // OR of step flags : redundant ? unless want to try to live without seqhis
    //

    unsigned long long* history = m_history->getValues() + 2*m_record_id ;
    *(history+0) = m_seqhis ; 
    *(history+1) = m_seqmat ; 

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

void Recorder::Dump(const char* msg, unsigned int index, const G4StepPoint* point, G4OpBoundaryProcessStatus boundary_status )
{
    std::string bs = OpBoundaryAbbrevString(boundary_status) ;
    std::cout << std::setw(7) << index << " " << Format(point, bs.c_str()) << std::endl ;
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
       Dump(msg, i, m_points[i], bst );
       //std::cout << std::hex << seqhis << std::dec << std::endl ; 
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



