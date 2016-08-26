#pragma once

class GItemIndex ; 

class Opticks ; 
class OpticksHub ; 
class OpticksEvent ; 

#include "OKGEO_API_EXPORT.hh"

class OKGEO_API OpticksIdx {
   public:
       OpticksIdx(OpticksHub* hub);
       void setEvent(OpticksEvent* evt);
   public:
       GItemIndex* makeHistoryItemIndex();
       GItemIndex* makeMaterialItemIndex();
       GItemIndex* makeBoundaryItemIndex();
   public:
       // hostside indexing 
       void indexEvtOld();
       void indexBoundariesHost();
   private:
        OpticksHub*    m_hub ; 
        Opticks*       m_opticks ; 
        OpticksEvent*  m_evt ; 

};



 
