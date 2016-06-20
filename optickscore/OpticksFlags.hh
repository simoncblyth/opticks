#pragma once

#include <string>
#include <map>

#include "OpticksPhoton.h"

//class Opticks ; 
//class OpticksAttrSeq ; 

class Index ; 

// replacing part of Types
// hmm this should be low level, but needing to pass in Opticks makes it not so


#include "OKCORE_API_EXPORT.hh"
class OKCORE_API OpticksFlags {
    public:
       static const char* ENUM_HEADER_PATH ;  
    public:
       static const char* ZERO_ ;
       static const char* CERENKOV_ ;
       static const char* SCINTILLATION_ ;

       static const char* MISS_ ;
       static const char* BULK_ABSORB_ ;
       static const char* BULK_REEMIT_ ;
       static const char* BULK_SCATTER_ ;
       static const char* SURFACE_DETECT_ ;
       static const char* SURFACE_ABSORB_ ;
       static const char* SURFACE_DREFLECT_ ;
       static const char* SURFACE_SREFLECT_ ;
       static const char* BOUNDARY_REFLECT_ ;
       static const char* BOUNDARY_TRANSMIT_ ;
       static const char* TORCH_ ;
       static const char* G4GUN_ ;   

       static const char* NAN_ABORT_ ;
       static const char* BAD_FLAG_ ;
       static const char* OTHER_ ;

       static const char* cerenkov_ ;
       static const char* scintillation_ ;
       static const char* torch_ ;
       static const char* g4gun_ ;
       static const char* other_ ;
    public:
       static const char* SourceType(int code);
       static const char* SourceTypeLowercase(int code);
       static unsigned int SourceCode(const char* type);
    public:
       static const char* Flag(const unsigned int flag);
       static std::string FlagSequence(const unsigned long long seqhis);
    public:
    //    OpticksFlags(Opticks* cache, const char* path=ENUM_HEADER_PATH);
        OpticksFlags(const char* path=ENUM_HEADER_PATH);
        void save(const char* idpath);
    private:
        void init(const char* path);
        Index* parseFlags(const char* path);
    public:
//        std::map<unsigned int, std::string> getNamesMap(); 
    public:
        Index*             getIndex();  
//        OpticksAttrSeq*    getAttrIndex();  
    private:
//        Opticks*           m_cache  ;
//        OpticksAttrSeq*    m_aindex ; 
        Index*             m_index ; 
};

 
