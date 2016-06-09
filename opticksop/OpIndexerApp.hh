#pragma once
#include <cstddef>

// optickscore-
class Opticks ; 
class OpticksEvent ; 
template <typename T> class OpticksCfg ;

// opticksop-
class OpIndexer ; 


class OpIndexerApp {
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


inline OpIndexerApp::OpIndexerApp(int argc, char** argv) 
   :   
     m_argc(argc),
     m_argv(argv),
     m_opticks(NULL),
     m_cfg(NULL),
     m_evt(NULL),
     m_indexer(NULL)
{
    init();
}



