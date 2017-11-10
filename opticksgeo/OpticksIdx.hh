#pragma once

class GItemIndex ; 

class Opticks ; 
class OpticksHub ; 
class OpticksRun ; 
class OpticksEvent ; 

#include "OKGEO_API_EXPORT.hh"

/**
OpticksIdx
===========

Wrapper around indexing functionality


*/


class OKGEO_API OpticksIdx {
   public:
       OpticksIdx(OpticksHub* hub);
   public:
       // presentation prep
       GItemIndex* makeHistoryItemIndex();
       GItemIndex* makeMaterialItemIndex();
       GItemIndex* makeBoundaryItemIndex();
   public:
       // hostside indexing 
       void indexEvtOld();
       void indexBoundariesHost();
       void indexSeqHost();
   private:
        // from OpticksRun, uncached
        OpticksEvent* getEvent();
   private:
        OpticksHub*    m_hub ; 
        Opticks*       m_ok ; 
        OpticksRun*    m_run ; 

};



 
