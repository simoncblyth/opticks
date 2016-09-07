#pragma once

#include <cstddef>
#include <string>

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class OpticksEventSpec ; 

class OKCORE_API OpticksEventSpec {
   public:
        static const char* OK_ ; 
        static const char* G4_ ; 
        static const char* NO_ ; 
   public:
        OpticksEventSpec(OpticksEventSpec* spec);
        OpticksEventSpec(const char* typ, const char* tag, const char* det, const char* cat=NULL);
        OpticksEventSpec* clone(unsigned tagoffset=0);   // non-zero tagoffset increments if +ve, and decrements if -ve
        void Summary(const char* msg="OpticksEventSpec::Summary");
        std::string brief();
        bool isG4();
        bool isOK();
        const char*  getEngine();
   private:
        void init();
   public:
        const char*  getTyp();
        const char*  getTag();
        const char*  getDet();
        const char*  getCat();
        const char*  getUDet();
        const char*  getDir();
   public:
        int          getITag();
   protected:
        const char*  m_typ ; 
        const char*  m_tag ; 
        const char*  m_det ; 
        const char*  m_cat ; 
   private:
        const char*  m_dir ; 
        int          m_itag ; 
};

#include "OKCORE_TAIL.hh"


