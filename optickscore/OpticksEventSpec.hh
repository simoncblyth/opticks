#pragma once

#include <cstddef>
#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class OpticksEventSpec ; 

class OKCORE_API OpticksEventSpec {
   public:
        OpticksEventSpec(OpticksEventSpec* spec);
        OpticksEventSpec(const char* typ, const char* tag, const char* det, const char* cat=NULL);
        void Summary(const char* msg="OpticksEventSpec::Summary");
   private:
        void init();
   public:
        const char*  getTyp();
        const char*  getTag();
        const char*  getDet();
        const char*  getCat();
        const char*  getUDet();
        const char*  getDir();
   protected:
        const char*  m_typ ; 
        const char*  m_tag ; 
        const char*  m_det ; 
        const char*  m_cat ; 
   private:
        const char*  m_dir ; 
};

#include "OKCORE_TAIL.hh"


