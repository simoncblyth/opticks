#pragma once

#include <string>

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class Opticks ; 
template <typename T> class OpticksCfg ;

class OKCORE_API OpticksEntry 
{
   private:
        static const char* GENERATE_ ; 
        static const char* TRIVIAL_ ; 
        static const char* NOTHING_ ; 
        static const char* SEEDTEST_ ; 
        static const char* TRACETEST_ ; 
        static const char* UNKNOWN_ ; 
   public:
        static const char*  Name(char code);
        static char CodeFromConfig(OpticksCfg<Opticks>* cfg);  
   public:
        OpticksEntry(unsigned index, char code);
   public:
        unsigned      getIndex();
        const char*   getName();
        std::string   description();
        bool          isTrivial();
        bool          isNothing();
        bool          isTraceTest();
   private:
        unsigned             m_index ; 
        char                 m_code ; 

};

#include "OKCORE_TAIL.hh"


