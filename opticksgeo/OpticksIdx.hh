#pragma once

class GItemIndex ; 

class Opticks ; 
class OpticksHub ; 
class OpticksRun ; 
class OpticksEvent ; 

class OpticksAttrSeq ;

#include "OKGEO_API_EXPORT.hh"

/**
OpticksIdx
===========

Wrapper around hostside(only?) indexing functionality


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
       // used for GUI seqmat and boundaries presentation
       OpticksAttrSeq*  getMaterialNames();
       OpticksAttrSeq*  getBoundaryNames();
       std::map<unsigned int, std::string> getBoundaryNamesMap();

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



 
