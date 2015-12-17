#pragma once

#include <cstring>

class Opticks ; 
template <typename T> class OpticksCfg ;
class GCache ; 
class GBndLib ;
class GGeoTestConfig ; 
class TorchStepNPY ;
class Detector ; 
class Recorder ; 
class G4RunManager ; 

class CfG4 
{
   public:
        CfG4(const char* prefix);
        virtual ~CfG4();
   private:
        void init();
   public:
        void configure(int argc, char** argv);
        void propagate();
        void save();
   private:
        const char*           m_prefix ;
        Opticks*              m_opticks ;  
        OpticksCfg<Opticks>*  m_cfg ;
        GGeoTestConfig*       m_testconfig ; 
        GCache*               m_cache ; 
        TorchStepNPY*         m_torch ; 
   private:
        Detector*             m_detector ; 
        Recorder*             m_recorder ; 
        G4RunManager*         m_runManager ;
   private:
        unsigned int          m_g4_nevt ; 
        unsigned int          m_g4_photons_per_event ; 
        unsigned int          m_num_photons ; 

};

inline CfG4::CfG4(const char* prefix) 
   :
     m_prefix(strdup(prefix)),
     m_opticks(NULL),
     m_cfg(NULL),
     m_testconfig(NULL),
     m_cache(NULL),
     m_torch(NULL),
     m_detector(NULL),
     m_recorder(NULL),
     m_runManager(NULL),
     m_g4_nevt(0),
     m_g4_photons_per_event(0),
     m_num_photons(0)
{
    init();
}


