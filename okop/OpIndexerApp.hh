#pragma once

/**
OpIndexerApp
===============

For standalone indexing tests ?

**/


class Opticks ;   // okc-
class OpticksEvent ; 
template <typename T> class OpticksCfg ;

class OpticksHub ;   // okg-
class OpticksRun ; 

class OContext ;   // optixrap-
class OScene ; 
class OEvent ; 

class OpIndexer ;   // opop-

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

      OScene*               m_scene ; 
      OContext*             m_ocontext ; 
      OEvent*               m_oevt ; 

      OpIndexer*            m_indexer ; 

};


