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
        OpticksEventSpec* clone(unsigned tagoffset=0) const ;   // non-zero tagoffset increments if +ve, and decrements if -ve
        void Summary(const char* msg="OpticksEventSpec::Summary") const ;
        std::string brief() const ;
        bool isG4() const ;
        bool isOK() const ;
        const char*  getEngine() const ;
   private:
        void init();
   public:
        const char*  getTyp() const ;
        const char*  getTag() const ;
        const char*  getDet() const ;
        const char*  getCat() const ;
        const char*  getUDet() const ;
        const char*  getDir() const ;
        const char*  getRelDir() const ; // without the base, ie returns directory portion starting "evt/"
        const char*  getFold() const ;   // one level above Dir without the tag 
   public:
        int          getITag() const ;
   protected:
        const char*  m_typ ; 
        const char*  m_tag ; 
        const char*  m_det ; 
        const char*  m_cat ; 
        const char*  m_udet ; 
   private:
        const char*  m_dir ; 
        const char*  m_reldir ; 
        const char*  m_fold ; 
        int          m_itag ; 
};

#include "OKCORE_TAIL.hh"


