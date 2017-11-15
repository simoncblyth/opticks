#pragma once

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class G4StepPoint ;


// npy-
template <typename T> class NPY ;

// okc-
class Opticks ; 
class OpticksEvent ; 

class CG4 ; 
struct CG4Ctx ; 
struct CPhoton ; 


class CFG4_API CWriter 
{
        friend class CRecorder ; 
    public: 
        // TODO: move into sysrap- 
         static short shortnorm( float v, float center, float extent );
         static unsigned char my__float2uint_rn( float f );
    public:
        CWriter(CG4* g4, bool dynamic);        
        void writeStepPoint(unsigned target_record_id, unsigned slot, const G4StepPoint* point, unsigned int flag, unsigned int material, const char* /*label*/ ) ;
        void addDynamicRecords();
        void writePhoton(const G4StepPoint* point, const CPhoton& photon );
        // *writePhoton* overwrites prior entries for REJOIN updating target_record_id 
   private:
        void initEvent(OpticksEvent* evt);
    private:

        CG4*               m_g4 ; 
        bool               m_dynamic ; 
        CG4Ctx&            m_ctx ; 
        Opticks*           m_ok ; 

        OpticksEvent*      m_evt ; 

        NPY<float>*               m_primary ; 

        NPY<short>*               m_records ; 
        NPY<float>*               m_photons ; 
        NPY<unsigned long long>*  m_history ; 

        NPY<short>*               m_dynamic_records ; 
        NPY<float>*               m_dynamic_photons ; 
        NPY<unsigned long long>*  m_dynamic_history ; 

        NPY<short>*               m_target_records ; 

        unsigned           m_verbosity ; 




};

#include "CFG4_TAIL.hh"


