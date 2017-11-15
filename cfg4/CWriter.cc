#include <cassert>


#include "G4StepPoint.hh"
#include "G4StepStatus.hh"
#include "G4ThreeVector.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"



#include "BBit.hh"

#include "CRecorder.h"
#include "CG4Ctx.hh"
#include "CPhoton.hh"
#include "CG4.hh"

#include "NGLM.hpp"
#include "NPY.hpp"

#include "OpticksFlags.hh"
#include "OpticksEvent.hh"

#include "CWriter.hh"

#include "PLOG.hh"


// TODO: move the statics into sysrap-


#define fitsInShort(x) !(((((x) & 0xffff8000) >> 15) + 1) & 0x1fffe)
#define iround(x) ((x)>=0?(int)((x)+0.5):(int)((x)-0.5))

short CWriter::shortnorm( float v, float center, float extent )  // static 
{
    // range of short is -32768 to 32767
    // Expect no positions out of range, as constrained by the geometry are bouncing on,
    // but getting times beyond the range eg 0.:100 ns is expected
    //  
    int inorm = iround(32767.0f * (v - center)/extent ) ;    // linear scaling into -1.f:1.f * float(SHRT_MAX)
    return fitsInShort(inorm) ? short(inorm) : SHRT_MIN  ;
} 

unsigned char CWriter::my__float2uint_rn( float f ) // static
{
    return iround(f);
}


CWriter::CWriter(CG4* g4, bool dynamic)
   :
   m_g4(g4),
   m_dynamic(dynamic),
   m_ctx(g4->getCtx()),
   m_ok(g4->getOpticks()),

   m_evt(NULL),

   m_primary(NULL),

   m_records(NULL),
   m_photons(NULL),
   m_history(NULL),

   m_dynamic_records(NULL),
   m_dynamic_photons(NULL),
   m_dynamic_history(NULL)
{
}

void CWriter::initEvent(OpticksEvent* evt)  // called by CRecorder::initEvent/CG4::initEvent
{
    m_evt = evt ; 
    assert(m_evt && m_evt->isG4());


    LOG(info) << "CWriter::initEvent"
              << " dynamic " << ( m_dynamic ? "DYNAMIC(CPU style)" : "STATIC(GPU style)" )
              << " record_max " << m_ctx._record_max
              << " bounce_max  " << m_ctx._bounce_max 
              << " steps_per_photon " << m_ctx._steps_per_photon 
              << " num_g4event " << m_evt->getNumG4Event() 
              ;

    if(m_dynamic)
    {
        assert(m_ctx._record_max == 0 );

        // shapes must match OpticksEvent::createBuffers
        // TODO: avoid this duplicity using the spec

        m_dynamic_records = NPY<short>::make(1, m_ctx._steps_per_photon, 2, 4) ;
        m_dynamic_records->zero();

        m_dynamic_photons = NPY<float>::make(1, 4, 4) ;
        m_dynamic_photons->zero();

        m_dynamic_history = NPY<unsigned long long>::make(1, 1, 2) ;
        m_dynamic_history->zero();
    } 
    else
    {
        assert(m_ctx._record_max > 0 );
    }

    m_history = m_evt->getSequenceData();
    m_photons = m_evt->getPhotonData();
    m_records = m_evt->getRecordData();

    m_target_records = m_dynamic ? m_dynamic_records : m_records ; 

    assert( m_history && "CRecorder requires history buffer" );
    assert( m_photons && "CRecorder requires photons buffer" );
    assert( m_records && "CRecorder requires records buffer" );
}


void CWriter::addDynamicRecords()
{
    m_records->add(m_dynamic_records);
}

void CWriter::writeStepPoint(unsigned target_record_id, unsigned slot, const G4StepPoint* point, unsigned int flag, unsigned int material, const char* /*label*/ )
{
    // write compressed record quads into buffer at location for the m_record_id 

    if(m_ctx._dbgrec)
    {
        LOG(info) << "[--dbgrec]"
                  << " target_record_id " << target_record_id
                  << " slot " << slot 
                  << " flag " << flag 
                  << " material " << material 
                  ;
    }  


    const G4ThreeVector& pos = point->GetPosition();
    const G4ThreeVector& pol = point->GetPolarization();

    G4double time = point->GetGlobalTime();
    G4double energy = point->GetKineticEnergy();
    G4double wavelength = h_Planck*c_light/energy ;

    const glm::vec4& sd = m_evt->getSpaceDomain() ; 
    const glm::vec4& td = m_evt->getTimeDomain() ; 
    const glm::vec4& wd = m_evt->getWavelengthDomain() ; 

    short posx = shortnorm(pos.x()/mm, sd.x, sd.w ); 
    short posy = shortnorm(pos.y()/mm, sd.y, sd.w ); 
    short posz = shortnorm(pos.z()/mm, sd.z, sd.w ); 
    short time_ = shortnorm(time/ns,   td.x, td.y );

    float wfrac = ((wavelength/nm) - wd.x)/wd.w ;   

    // see oxrap/cu/photon.h
    unsigned char polx = my__float2uint_rn( (pol.x()+1.f)*127.f );
    unsigned char poly = my__float2uint_rn( (pol.y()+1.f)*127.f );
    unsigned char polz = my__float2uint_rn( (pol.z()+1.f)*127.f );
    unsigned char wavl = my__float2uint_rn( wfrac*255.f );

/*
    LOG(info) << "CWriter::RecordStepPoint"
              << " wavelength/nm " << wavelength/nm 
              << " wd.x " << wd.x
              << " wd.w " << wd.w
              << " wfrac " << wfrac 
              << " wavl " << unsigned(wavl) 
              ;
*/

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

    //unsigned int target_record_id = m_dynamic ? 0 : m_record_id ; 

    m_target_records->setQuad(target_record_id, slot, 0, posx, posy, posz, time_ );
    m_target_records->setQuad(target_record_id, slot, 1, polw.short_.x, polw.short_.y, polw.short_.z, polw.short_.w );  

    // dynamic mode : fills in slots into single photon dynamic_records structure 
    // static mode  : fills directly into a large fixed dimension records structure

    // looks like static mode will succeed to scrub the AB and replace with RE 
    // just by decrementing m_slot and running again
    // but dynamic mode will have an extra record
}

void CWriter::writePhoton(const G4StepPoint* point, const CPhoton& photon )
{
    // gets called at last step (eg absorption) or when truncated
    // for reemission have to rely on downstream overwrites
    // via rerunning with a target_record_id to scrub old values


    const G4ThreeVector& pos = point->GetPosition();
    const G4ThreeVector& dir = point->GetMomentumDirection();
    const G4ThreeVector& pol = point->GetPolarization();

    G4double time = point->GetGlobalTime();
    G4double energy = point->GetKineticEnergy();
    G4double wavelength = h_Planck*c_light/energy ;
    G4double weight = 1.0 ; 

    NPY<float>* target = m_dynamic ? m_dynamic_photons : m_photons ; 
    unsigned int target_record_id = m_dynamic ? 0 : m_ctx._record_id ; 


    target->setQuad(target_record_id, 0, 0, pos.x()/mm, pos.y()/mm, pos.z()/mm, time/ns  );
    target->setQuad(target_record_id, 1, 0, dir.x(), dir.y(), dir.z(), weight  );
    target->setQuad(target_record_id, 2, 0, pol.x(), pol.y(), pol.z(), wavelength/nm  );

    target->setUInt(target_record_id, 3, 0, 0, photon._slot );
    target->setUInt(target_record_id, 3, 0, 1, 0u );
    target->setUInt(target_record_id, 3, 0, 2, photon._c4.u );
    target->setUInt(target_record_id, 3, 0, 3, photon._mskhis );

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
    *(history+0) = photon._seqhis ; 
    *(history+1) = photon._seqmat ; 

    if(m_dynamic)
    {
        m_history->add(m_dynamic_history);
    }
}



