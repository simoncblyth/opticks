#include "Recorder.hh"
#include "Format.hh"

#include "Opticks.hh"
#include "OpticksFlags.h"

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

// cfg4-
#include "Recorder.h"
#include "CPropLib.hh"



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
    m_c4.u = 0u ; 

    m_record_max = m_evt->getNumPhotons(); 
    m_bounce_max = m_evt->getBounceMax();
    m_steps_per_photon = m_evt->getMaxRec() ;    

    m_step = m_evt->isStep();

    LOG(info) << "Recorder::init"
              << " record_max " << m_record_max
              << " bounce_max  " << m_bounce_max 
              << " steps_per_photon " << m_steps_per_photon 
              << " isStep " << m_step  
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



void Recorder::startPhoton()
{
    //LOG(info) << "Recorder::startPhoton" ; 

    if(m_record_id % 10000 == 0) Summary("Recorder::startPhoton") ;

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


void Recorder::setBoundaryStatus(G4OpBoundaryProcessStatus boundary_status, unsigned int premat, unsigned int postmat)
{
    // this is invoked before RecordStep is called from SteppingAction
    m_prior_boundary_status = m_boundary_status ; 
    m_prior_premat = m_premat ; 
    m_prior_postmat = m_postmat ; 

    m_boundary_status = boundary_status ; 
    m_premat = premat ; 
    m_postmat = postmat ; 

}



/*

[2016-Mar-01 16:56:59.319217]:info: Recorder::RecordStep  step_id 0 pre            Undefined post         GeomBoundary
[2016-Mar-01 16:56:59.319367]:info: Recorder::RecordStep  step_id 1 pre         GeomBoundary post         GeomBoundary
[2016-Mar-01 16:56:59.319503]:info: Recorder::RecordStep  step_id 2 pre         GeomBoundary post        WorldBoundary
[2016-Mar-01 16:56:59.319617]:info: SteppingAction::UserSteppingAction DONE record_id   17516 seqhis 8ccd TORCH BOUNDARY_TRANSMIT BOUNDARY_TRANSMIT SURFACE_ABSORB . . . . . . . . . . . .  seqmat 14e4 13 12 12 8 0 0 0 0 0 0 0 0 0 0 0 0 
      0 Und       box_phys         NoProc           Undefined pos[     2.48   -97.3     300]  dir[        0       0      -1]  pol[       -1 -0.0254       0]  ns  0.100 nm 380.000
      1 FrT      pvPmtHemi Transportation        GeomBoundary pos[     2.48   -97.3    73.4]  dir[  0.00152 -0.0598  -0.998]  pol[       -1 -0.0254-6.65e-18]  ns  1.295 nm 380.000
      2 FrT       box_phys Transportation        GeomBoundary pos[     2.52     -99    44.7]  dir[  0.00278  -0.109  -0.994]  pol[       -1 -0.02542.77e-17]  ns  1.435 nm 380.000
      3 FrT                Transportation       WorldBoundary pos[     3.49    -137    -300]  dir[  0.00278  -0.109  -0.994]  pol[       -1 -0.02542.77e-17]  ns  3.265 nm 380.000
*/



/*
  Mapping G4Step/G4StepPoint into Opticks records style is the point of *Recorder*


 Opticks records...

       flag set to generation code
       while(bounce < bounce_max)
       {
             rtTrace(..)     // look ahead to get boundary 

             fill_state()    // interpret boundary into m1 m2 material codes based on photon direction

             RSAVE()         

                             // change photon position/direction/time/flag 
             propagate_to_boundary
             propagate_at_surface or at_boundary   

             break/continue depending on flag
       }

       RSAVE()


  Consider reflection

     G4:
         G4Step at boundary straddles the volumes, with post step point 
         in a volume that will not be entered. 

         Schematic of the steps and points of a reflection


        step1    step2    step3


          *      .   .
           \     .   .
            \    .   .
             \   .   . 
              \  .   .  
               * . * .
                 . * .  *
                 .   .   \
                 .   .    \
                 .   .     \
                 .   .      * 
                 .   . 


               



     Op:
         m2 gets set to the material that will not be entered by the lookahead
         m1 always stays the material that photon is in 

*/



bool Recorder::RecordStep(const G4Step* step)
{
    const G4StepPoint* pre  = step->GetPreStepPoint() ; 
    const G4StepPoint* post = step->GetPostStepPoint() ; 

    const G4Material* preMat  = pre->GetMaterial() ;
    const G4Material* postMat = post->GetMaterial() ;

    /*
    if(m_debug)
    {
       LOG(info) << "Recorder::RecordStep " 
                 << " step_id " << m_step_id
                 << " pre " << std::setw(20) << OpStepString(pre->GetStepStatus()) 
                 << " post " << std::setw(20) << OpStepString(post->GetStepStatus()) 
                 << " preMat " << std::setw(4) << preMat << std::setw(20)  << ( preMat == 0 ? "-" : m_clib->getMaterialName(preMat-1)  )
                 << " postMat " << std::setw(4) << postMat << std::setw(20) << ( postMat == 0 ? "-" : m_clib->getMaterialName(postMat-1)  )
                 << " preM " <<  ( preM ?  preM->GetName() : "nul" )
                 << " postM " << ( postM ? postM->GetName() : "nul" )
                 ;
    }
    */

    unsigned int preFlag ; 
    unsigned int postFlag ; 

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

    bool absorb   = (postFlag & (BULK_ABSORB | SURFACE_ABSORB | SURFACE_DETECT)) != 0 ;

    bool preSkip = m_prior_boundary_status == StepTooSmall ;  

    bool done = false ; 


    // skip the pre, but the post becomes the pre at next step where will be taken 
    // 1-based material indices, so zero can represent None
    if(!preSkip)
    {
       done = RecordStepPoint( pre, preFlag, m_prior_premat, m_prior_boundary_status, PRE ); 
    }

    if(absorb && !done)
    {
       done = RecordStepPoint( post, postFlag, m_postmat, m_boundary_status, POST ); 
    }

    // when not *absorb* the post step will become the pre step at next RecordStep

    return done ;
}






bool Recorder::RecordStepPoint(const G4StepPoint* point, unsigned int flag, unsigned int material, G4OpBoundaryProcessStatus boundary_status, const char* label)
{
    bool absorb = ( flag & (BULK_ABSORB | SURFACE_ABSORB | SURFACE_DETECT)) != 0 ;

    unsigned int slot =  m_slot < m_steps_per_photon  ? m_slot : m_steps_per_photon - 1 ;

    //Dump(label,  slot, point, boundary_status );

    if(m_step)
    {
        unsigned long long shift = slot*4ull ;   
        unsigned long long msk = 0xFull << shift ; 
        unsigned long long his = ffs(flag) & 0xFull ; 
        unsigned long long mat = material < 0xFull ? material : 0xFull ; 
        m_seqhis =  (m_seqhis & (~msk)) | (his << shift) ; 
        m_seqmat =  (m_seqmat & (~msk)) | (mat << shift) ; 

        RecordStepPoint(slot, point, flag, material, label);

        if(m_debug) Collect(point, flag, material, boundary_status, m_seqhis, m_seqmat);
    }

    m_slot += 1 ; 

    bool truncate = m_slot > m_bounce_max  ;  

    return truncate || absorb ;
}


void Recorder::RecordStepPoint(unsigned int slot, const G4StepPoint* point, unsigned int flag, unsigned int material, const char* label )
{
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
    qaux.uchar_.x = material ; 
    qaux.uchar_.y = 0 ; // TODO:m2 
    qaux.char_.z  = 0 ; // TODO:boundary (G4 equivalent ?)
    qaux.uchar_.w = ffs(flag) ;   // ? duplicates seqhis  

    hquad polw ; 
    polw.ushort_.x = polx | poly << 8 ; 
    polw.ushort_.y = polz | wavl << 8 ; 
    polw.ushort_.z = qaux.uchar_.x | qaux.uchar_.y << 8  ;
    polw.ushort_.w = qaux.uchar_.z | qaux.uchar_.w << 8  ;

    m_records->setQuad(m_record_id, slot, 1, polw.short_.x, polw.short_.y, polw.short_.z, polw.short_.w );  
}










void Recorder::RecordQuadrant(const G4Step* step)
{
    const G4StepPoint* pre  = step->GetPreStepPoint() ; 
    const G4ThreeVector& pos = pre->GetPosition();

    // initial quadrant 
    m_c4.uchar_.x = 
                  (  pos.x() > 0.f ? QX : 0u ) 
                   |   
                  (  pos.y() > 0.f ? QY : 0u ) 
                   |   
                  (  pos.z() > 0.f ? QZ : 0u )
                  ;   

    m_c4.uchar_.y = 2u ; 
    m_c4.uchar_.z = 3u ; 
    m_c4.uchar_.w = 4u ; 
}



void Recorder::RecordPhoton(const G4Step* step)
{
    const G4StepPoint* point  = step->GetPostStepPoint() ; 

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

    m_photons->setUInt(m_record_id, 3, 0, 0, m_slot );
    m_photons->setUInt(m_record_id, 3, 0, 1, 0u );
    m_photons->setUInt(m_record_id, 3, 0, 2, m_c4.u );
    m_photons->setUInt(m_record_id, 3, 0, 3, 0u );

    // generate.cu
    //
    //  (x)  p.flags.i.x = prd.boundary ;   // last boundary
    //  (y)  p.flags.u.y = s.identity.w ;   // sensorIndex  >0 only for cathode hits
    //  (z)  p.flags.u.z = s.index.x ;      // material1 index  : redundant with boundary  
    //  (w)  p.flags.u.w |= s.flag ;        // OR of step flags : redundant ? unless want to try to live without seqhis
    //

    if(m_step)
    {
        unsigned long long* history = m_history->getValues() + 2*m_record_id ;
        *(history+0) = m_seqhis ; 
        *(history+1) = m_seqmat ; 
    }
}







bool Recorder::hasIssue()
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

void Recorder::Dump(const char* msg, unsigned int index, const G4StepPoint* point, G4OpBoundaryProcessStatus boundary_status, const char* matname )
{
    std::string bs = OpBoundaryAbbrevString(boundary_status) ;
    std::cout << std::setw(7) << index << " " << std::setw(15) << matname << " " << Format(point, bs.c_str()) << std::endl ;
}

void Recorder::Dump(const char* msg)
{
    LOG(info) << msg 
              << " record_id " << std::setw(7) << m_record_id
              << std::endl 
              << " seqhis " << std::hex << m_seqhis << std::dec 
              << " " << Opticks::FlagSequence(m_seqhis) 
              << std::endl 
              << " seqmat " << std::hex << m_seqmat << std::dec 
              << " " << m_clib->MaterialSequence(m_seqmat) 
              ;

    if(!m_debug) return ; 

    for(unsigned int i=0 ; i<m_points.size() ; i++) 
    {
       unsigned long long seqhis = m_seqhis_dbg[i] ;
       unsigned long long seqmat = m_seqmat_dbg[i] ;
       G4OpBoundaryProcessStatus bst = m_bndstats[i] ;
       unsigned int mat = m_materials[i] ;
       const char* matname = ( mat == 0 ? "-" : m_clib->getMaterialName(mat-1)  ) ;

       Dump(msg, i, m_points[i], bst, matname );

       //std::cout << std::hex << seqhis << std::dec << std::endl ; 
       //std::cout << std::hex << seqmat << std::dec << std::endl ; 
    }
}




void Recorder::Collect(const G4StepPoint* point, unsigned int flag, unsigned int material, G4OpBoundaryProcessStatus boundary_status, unsigned long long seqhis, unsigned long long seqmat)
{
    assert(m_debug);
    m_points.push_back(new G4StepPoint(*point));
    m_flags.push_back(flag);
    m_materials.push_back(material);
    m_bndstats.push_back(boundary_status);  // will duplicate the status for the last step
    m_seqhis_dbg.push_back(seqhis);
    m_seqmat_dbg.push_back(seqmat);
}

void Recorder::Clear()
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

    m_primary->setUInt(m_primary_id, 3, 0, 0, 0u );
    m_primary->setUInt(m_primary_id, 3, 0, 1, 0u );
    m_primary->setUInt(m_primary_id, 3, 0, 2, 0u );
    m_primary->setUInt(m_primary_id, 3, 0, 3, 0u );

    m_primary_id += 1 ; 

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



