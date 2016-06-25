#pragma once

// optickscore-
class Opticks ; 
class OpticksEvent ; 
template <typename T> class OpticksCfg ;

// opticksop-
class OpIndexer ; 

#include "OKOP_API_EXPORT.hh"
class OKOP_API OpIndexerApp {
   public:
      OpIndexerApp(int argc, char** argv);
      void configure();
      void loadEvtFromFile(bool verbose=false);
      void makeIndex();
   private:
      void init();
   private:
      int                   m_argc ; 
      char**                m_argv ; 
      Opticks*              m_opticks ;   
      OpticksCfg<Opticks>*  m_cfg ;
      OpticksEvent*         m_evt ;
      OpIndexer*            m_indexer ; 

};


