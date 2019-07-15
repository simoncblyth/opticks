#include <cassert>


#include "G4StepPoint.hh"
#include "G4StepStatus.hh"
#include "G4ThreeVector.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "Randomize.hh"


#include "BConverter.hh"
#include "BBit.hh"

#include "CRecorder.h"
#include "CG4Ctx.hh"
#include "CPhoton.hh"
#include "Opticks.hh"
#include "CG4.hh"

#include "NGLM.hpp"
#include "NPY.hpp"

#include "OpticksFlags.hh"
#include "OpticksEvent.hh"

#include "CWriter.hh"
#include "PLOG.hh"

const plog::Severity CWriter::LEVEL = PLOG::EnvLevel("CWriter", "DEBUG") ; 


CWriter::CWriter(CG4* g4, CPhoton& photon, bool dynamic)
    :
    m_g4(g4),
    m_photon(photon),
    m_dynamic(dynamic),
    m_ctx(g4->getCtx()),
    m_ok(g4->getOpticks()),
    m_enabled(true),

    m_evt(NULL),

    m_primary(NULL),

    m_records_buffer(NULL),
    m_photons_buffer(NULL),
    m_history_buffer(NULL),

    m_dynamic_records(NULL),
    m_dynamic_photons(NULL),
    m_dynamic_history(NULL)
{
    LOG(LEVEL) << " " << ( m_dynamic ? "DYNAMIC" : "STATIC" ) ;
}

void CWriter::setEnabled(bool enabled)
{
    m_enabled = enabled ; 
}

/**
CWriter::initEvent
-------------------

Gets refs to the history, photons and records buffers from the event.
When dynamic the records target is single item dynamic_records otherwise
goes direct to the records_buffer.

**/

void CWriter::initEvent(OpticksEvent* evt)  // called by CRecorder::initEvent/CG4::initEvent
{
    m_evt = evt ; 
    assert(m_evt && m_evt->isG4());

    m_evt->setDynamic( m_dynamic ? 1 : 0 ) ;  

    LOG(LEVEL) 
        << ( m_dynamic ? "DYNAMIC(CPU style)" : "STATIC(GPU style)" )
        << " _record_max " << m_ctx._record_max
        << " _bounce_max  " << m_ctx._bounce_max 
        << " _steps_per_photon " << m_ctx._steps_per_photon 
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

    m_history_buffer = m_evt->getSequenceData();
    m_photons_buffer = m_evt->getPhotonData();
    m_records_buffer = m_evt->getRecordData();

    m_target_records = m_dynamic ? m_dynamic_records : m_records_buffer ; 

    assert( m_history_buffer && "CRecorder requires history buffer" );
    assert( m_photons_buffer && "CRecorder requires photons buffer" );
    assert( m_records_buffer && "CRecorder requires records buffer" );
}


        

// invoked by CRecorder::RecordStepPoint
bool CWriter::writeStepPoint(const G4StepPoint* point, unsigned flag, unsigned material )
{
    m_photon.add(flag, material);

    bool hard_truncate = m_photon.is_hard_truncate();

    bool done = false ; 

    if(hard_truncate) 
    {
        done = true ; 
    }
    else
    {
        if(m_enabled) writeStepPoint_(point, m_photon );

        m_photon.increment_slot() ; 

        done = m_photon.is_done() ;  // caution truncation/is_done may change after increment

        if( done && m_enabled )
        {
            writePhoton(point);
            if(m_dynamic) m_records_buffer->add(m_dynamic_records);
        }
    }        

    if( flag == BULK_ABSORB )
    {
        assert( done == true ); 
    }

    return done ; 
}


/**
CWriter::writeStepPoint_
--------------------------

Writes compressed step records if Geant4 CG4 instrumented events 
in format to match that written on GPU by oxrap.

**/

void CWriter::writeStepPoint_(const G4StepPoint* point, const CPhoton& photon )
{
    // write compressed record quads into buffer at location for the m_record_id 

    unsigned target_record_id = m_dynamic ? 0 : m_ctx._record_id ; 
    unsigned slot = photon._slot_constrained ;
    unsigned flag = photon._flag ; 
    unsigned material = photon._mat ; 


    if(!m_dynamic) assert( target_record_id < m_ctx._record_max );

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
    //LOG(info) << " pos " <<  std::setw(30) << pos << " pol " <<  std::setw(30) << pol ; 

    G4double time = point->GetGlobalTime();
    G4double energy = point->GetKineticEnergy();
    G4double wavelength = h_Planck*c_light/energy ;

    const glm::vec4& sd = m_evt->getSpaceDomain() ; 
    const glm::vec4& td = m_evt->getTimeDomain() ; 
    const glm::vec4& wd = m_evt->getWavelengthDomain() ; 


    short posx = BConverter::shortnorm(pos.x()/mm, sd.x, sd.w ); 
    short posy = BConverter::shortnorm(pos.y()/mm, sd.y, sd.w ); 
    short posz = BConverter::shortnorm(pos.z()/mm, sd.z, sd.w ); 
    short time_ = BConverter::shortnorm(time/ns,   td.x, td.y );

    float wfrac = ((wavelength/nm) - wd.x)/wd.w ;   

    // see oxrap/cu/photon.h
    // tboolean-box : for first steppoint the pol is same as pos causing out-of-range
    // notes/issues/tboolean-resurrection.rst
    unsigned char polx = BConverter::my__float2uint_rn( (pol.x()+1.f)*127.f );
    unsigned char poly = BConverter::my__float2uint_rn( (pol.y()+1.f)*127.f );
    unsigned char polz = BConverter::my__float2uint_rn( (pol.z()+1.f)*127.f );
    unsigned char wavl = BConverter::my__float2uint_rn( wfrac*255.f );

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


/**
CWriter::writePhoton
--------------------

Gets called at last step (eg absorption) or when truncated,
for reemission have to rely on downstream overwrites
via rerunning with a target_record_id to scrub old values.

In static case (where the number of photons is known ahead of time) 
directly populates the pre-sized photon buffer in dynamic case populates 
the single item m_dynamic_photons buffer first and then adds that to 
the m_photons_buffer

generate.cu

(x)  p.flags.i.x = prd.boundary ;   // last boundary
(y)  p.flags.u.y = s.identity.w ;   // sensorIndex  >0 only for cathode hits
(z)  p.flags.u.z = s.index.x ;      // material1 index  : redundant with boundary  
(w)  p.flags.u.w |= s.flag ;        // OR of step flags : redundant ? unless want to try to live without seqhis

**/

void CWriter::writePhoton(const G4StepPoint* point )
{
    const G4ThreeVector& pos = point->GetPosition();
    const G4ThreeVector& dir = point->GetMomentumDirection();
    const G4ThreeVector& pol = point->GetPolarization();

    G4double time = point->GetGlobalTime();
    G4double energy = point->GetKineticEnergy();
    G4double wavelength = h_Planck*c_light/energy ;
    G4double weight = 1.0 ;  

    NPY<float>* target = m_dynamic ? m_dynamic_photons : m_photons_buffer ; 
    unsigned target_record_id = m_dynamic ? 0 : m_ctx._record_id ; 

    target->setQuad(target_record_id, 0, 0, pos.x()/mm, pos.y()/mm, pos.z()/mm, time/ns  );
    target->setQuad(target_record_id, 1, 0, dir.x(), dir.y(), dir.z(), weight  );
    target->setQuad(target_record_id, 2, 0, pol.x(), pol.y(), pol.z(), wavelength/nm  );

    target->setUInt(target_record_id, 3, 0, 0, m_photon._slot_constrained );

    if( m_ok->isUTailDebug() )     // --utaildebug
    { 
        G4double u = G4UniformRand() ; 
        target->setValue(target_record_id, 3, 0, 1, u );
    }
    else
    {
        target->setUInt(target_record_id, 3, 0, 1, 0u );
    }     

    target->setUInt(target_record_id, 3, 0, 2, m_photon._c4.u );
    target->setUInt(target_record_id, 3, 0, 3, m_photon._mskhis );

    if(m_dynamic)
    {
        m_photons_buffer->add(m_dynamic_photons);
    }

    NPY<unsigned long long>* h_target = m_dynamic ? m_dynamic_history : m_history_buffer ; 

    unsigned long long* history = h_target->getValues() + 2*target_record_id ;
    *(history+0) = m_photon._seqhis ; 
    *(history+1) = m_photon._seqmat ; 

    if(m_dynamic)
    {
        m_history_buffer->add(m_dynamic_history);
    }
}

