#pragma once

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class G4StepPoint ;

// okc-
class OpticksEvent ; 

// npy-
template <typename T> class NPY ;


class CFG4_API CRecorderWriter 
{
        friend class CRecorder ; 
    public: 
        // TODO: move into sysrap- 
         static short shortnorm( float v, float center, float extent );
         static unsigned char my__float2uint_rn( float f );
    public:
        CRecorderWriter();        
        void RecordStepPoint(unsigned target_record_id, unsigned slot, const G4StepPoint* point, unsigned int flag, unsigned int material, const char* /*label*/ ) ;
   private:
        void setEvent(OpticksEvent* evt);
        void setTarget(NPY<short>* target);
    private:
        OpticksEvent*      m_evt ; 
        NPY<short>*        m_target ; 

};

#include "CFG4_TAIL.hh"


