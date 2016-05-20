#pragma once

#include <cstdlib>

// optickscore-
class Opticks ; 
template <typename T> class OpticksCfg ;

// ggeo-
class GCache ; 
class GBndLib ;
class GGeoTestConfig ; 

// npy--
class TorchStepNPY ;
class NumpyEvt ; 

// cfg4-
class CG4 ; 
class CPropLib ; 

class CDetector ; 
class CSteppingAction ;
class CPrimaryGeneratorAction ;

class Recorder ; 
class Rec ; 


class CCfG4 
{
   public:
        CCfG4(int argc, char** argv);
        virtual ~CCfG4();
   private:
        void init(int argc, char** argv);
   private:
        void configure(int argc, char** argv);
        CDetector* configureDetector();
        CSteppingAction* configureStepping();
        CPrimaryGeneratorAction* configureGenerator();
   public:
        void interactive(int argc, char** argv);
        void propagate();
        void save();
   private:
        void setupDomains();
   private:
        CG4*                  m_geant4 ; 
        Opticks*              m_opticks ;  
        OpticksCfg<Opticks>*  m_cfg ;
        GCache*               m_cache ; 
        TorchStepNPY*         m_torch ; 
        NumpyEvt*             m_evt ; 
   private:
        CDetector*            m_detector ; 
        CPropLib*             m_lib ; 
        Recorder*             m_recorder ; 
        Rec*                  m_rec ; 
   private:
        unsigned int          m_num_g4event ; 
        unsigned int          m_num_photons ; 

};



inline CCfG4::CCfG4(int argc, char** argv) 
   :
     m_geant4(NULL),
     m_opticks(NULL),
     m_cfg(NULL),
     m_cache(NULL),
     m_torch(NULL),
     m_evt(NULL),
     m_detector(NULL),
     m_lib(NULL),
     m_recorder(NULL),
     m_rec(NULL),
     m_num_g4event(0),
     m_num_photons(0)
{
    init(argc, argv);
}


