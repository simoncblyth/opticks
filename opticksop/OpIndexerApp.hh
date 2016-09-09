#pragma once

// optickscore-
class Opticks ; 
class OpticksEvent ; 
template <typename T> class OpticksCfg ;

// opticksgeo-
class OpticksHub ; 
class OpticksRun ; 

// opticksop-
class OpEngine ; 
class OpIndexer ; 

#include "OKOP_API_EXPORT.hh"
class OKOP_API OpIndexerApp {
   public:
      OpIndexerApp(int argc, char** argv);
   public:
      void loadEvtFromFile();
      void makeIndex();
   private:
      Opticks*              m_ok ;   
      OpticksCfg<Opticks>*  m_cfg ;
      OpticksHub*           m_hub ;   
      OpticksRun*           m_run ;   

      OpEngine*             m_engine ; 
      OpIndexer*            m_indexer ; 

};


