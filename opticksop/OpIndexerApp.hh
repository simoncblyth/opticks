#pragma once

// optickscore-
class Opticks ; 
class OpticksEvent ; 
template <typename T> class OpticksCfg ;

// opticksgeo-
class OpticksHub ; 

// opticksop-
class OpIndexer ; 

#include "OKOP_API_EXPORT.hh"
class OKOP_API OpIndexerApp {
   public:
      OpIndexerApp(int argc, char** argv);
      void configure();
      void loadEvtFromFile();
      void makeIndex();
   private:
      void init();
   private:
      Opticks*              m_opticks ;   
      OpticksHub*           m_hub ;   
      OpticksCfg<Opticks>*  m_cfg ;
      OpIndexer*            m_indexer ; 

};


