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
        CWriter(CG4* g4, CPhoton& photon, bool dynamic);        

        void setEnabled(bool enabled);
        bool writeStepPoint(const G4StepPoint* point, unsigned flag, unsigned material );
        void writePhoton(const G4StepPoint* point );
        // *writePhoton* overwrites prior entries for REJOIN updating target_record_id 
   private:
        void writeStepPoint_(const G4StepPoint* point, const CPhoton& photon );
        void initEvent(OpticksEvent* evt);
    private:

        CG4*               m_g4 ; 
        CPhoton&           m_photon ; 
        bool               m_dynamic ; 
        CG4Ctx&            m_ctx ; 
        Opticks*           m_ok ; 
        bool               m_enabled ; 

        OpticksEvent*      m_evt ; 

        NPY<float>*               m_primary ; 

        NPY<short>*               m_records_buffer ; 
        NPY<float>*               m_photons_buffer ; 
        NPY<unsigned long long>*  m_history_buffer ; 

        NPY<short>*               m_dynamic_records ; 
        NPY<float>*               m_dynamic_photons ; 
        NPY<unsigned long long>*  m_dynamic_history ; 

        NPY<short>*               m_target_records ; 

        unsigned           m_verbosity ; 




};

#include "CFG4_TAIL.hh"


