#include <cassert>


#include "G4StepPoint.hh"
#include "G4StepStatus.hh"
#include "G4ThreeVector.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"



#include "BBit.hh"

#include "CRecorder.h"

#include "NGLM.hpp"
#include "NPY.hpp"

#include "OpticksEvent.hh"

#include "CRecorderWriter.hh"



// TODO: move the statics into sysrap-


#define fitsInShort(x) !(((((x) & 0xffff8000) >> 15) + 1) & 0x1fffe)
#define iround(x) ((x)>=0?(int)((x)+0.5):(int)((x)-0.5))

short CRecorderWriter::shortnorm( float v, float center, float extent )  // static 
{
    // range of short is -32768 to 32767
    // Expect no positions out of range, as constrained by the geometry are bouncing on,
    // but getting times beyond the range eg 0.:100 ns is expected
    //  
    int inorm = iround(32767.0f * (v - center)/extent ) ;    // linear scaling into -1.f:1.f * float(SHRT_MAX)
    return fitsInShort(inorm) ? short(inorm) : SHRT_MIN  ;
} 

unsigned char CRecorderWriter::my__float2uint_rn( float f ) // static
{
    return iround(f);
}





CRecorderWriter::CRecorderWriter()
   :
   m_evt(NULL)
{
}


void CRecorderWriter::setEvent(OpticksEvent* evt)
{
    m_evt = evt ; 
    assert(m_evt && m_evt->isG4());
}

void CRecorderWriter::setTarget(NPY<short>* target)
{
    m_target = target ; 
}




void CRecorderWriter::RecordStepPoint(unsigned target_record_id, unsigned slot, const G4StepPoint* point, unsigned int flag, unsigned int material, const char* /*label*/ )
{
    // write compressed record quads into buffer at location for the m_record_id 

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
    LOG(info) << "CRecorderWriter::RecordStepPoint"
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

    //NPY<short>* target = m_dynamic ? m_dynamic_records : m_records ; 
    //unsigned int target_record_id = m_dynamic ? 0 : m_record_id ; 

    m_target->setQuad(target_record_id, slot, 0, posx, posy, posz, time_ );
    m_target->setQuad(target_record_id, slot, 1, polw.short_.x, polw.short_.y, polw.short_.z, polw.short_.w );  

    // dynamic mode : fills in slots into single photon dynamic_records structure 
    // static mode  : fills directly into a large fixed dimension records structure

    // looks like static mode will succeed to scrub the AB and replace with RE 
    // just by decrementing m_slot and running again
    // but dynamic mode will have an extra record
}




