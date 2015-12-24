#pragma once
#include <cstddef>

class Opticks ; 
template <typename T> class OpticksCfg ;

class NLog ; 
class NumpyEvt ; 

class OpIndexer ; 


class OpIndexerApp {
   public:
      OpIndexerApp();
      void configure(int argc, char** argv);
      void loadEvtFromFile(bool verbose=false);
      void makeIndex();
   private:
      void init();
   private:
      NLog*                 m_log ; 
      Opticks*              m_opticks ;   
      OpticksCfg<Opticks>*  m_cfg ;
      NumpyEvt*             m_evt ; // TODO: migrated to Opticks and rename: OpticksEvt 
      OpIndexer*            m_indexer ; 

};


inline OpIndexerApp::OpIndexerApp() 
   :   
     m_log(NULL),
     m_opticks(NULL),
     m_cfg(NULL),
     m_evt(NULL),
     m_indexer(NULL)
{
    init();
}



