#pragma once

#include <cstddef>
#include <string>
#include <map>

class Opticks ; 
class OpticksAttrSeq ; 
class Index ; 

// replacing part of Types

class OpticksFlags {
    public:
        static const char* ENUM_HEADER_PATH ;  
    public:
        OpticksFlags(Opticks* cache, const char* path=ENUM_HEADER_PATH);
        void save(const char* idpath);
    private:
        void init(const char* path);
        Index* parseFlags(const char* path);
    public:
        std::map<unsigned int, std::string> getNamesMap(); 
    public:
        Index*      getIndex();  
        OpticksAttrSeq*   getAttrIndex();  
    private:
        Opticks*     m_cache  ;
        OpticksAttrSeq*    m_aindex ; 
        Index*       m_index ; 
};

inline OpticksFlags::OpticksFlags(Opticks* cache, const char* path) 
    :
    m_cache(cache),
    m_aindex(NULL),
    m_index(NULL)
{
    init(path);
}

inline OpticksAttrSeq* OpticksFlags::getAttrIndex()
{
    return m_aindex ; 
}  
inline Index* OpticksFlags::getIndex()
{
    return m_index ; 
}  
