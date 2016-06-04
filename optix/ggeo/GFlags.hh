#pragma once

#include <cstddef>
#include <string>
#include <map>

class Opticks ; 
class GAttrSeq ; 
class Index ; 

// replacing part of Types

class GFlags {
    public:
        static const char* ENUM_HEADER_PATH ;  
    public:
        GFlags(Opticks* cache, const char* path=ENUM_HEADER_PATH);
        void save(const char* idpath);
    private:
        void init(const char* path);
        Index* parseFlags(const char* path);
    public:
        std::map<unsigned int, std::string> getNamesMap(); 
    public:
        Index*      getIndex();  
        GAttrSeq*   getAttrIndex();  
    private:
        Opticks*     m_cache  ;
        GAttrSeq*    m_aindex ; 
        Index*       m_index ; 
};

inline GFlags::GFlags(Opticks* cache, const char* path) 
    :
    m_cache(cache),
    m_aindex(NULL),
    m_index(NULL)
{
    init(path);
}

inline GAttrSeq* GFlags::getAttrIndex()
{
    return m_aindex ; 
}  
inline Index* GFlags::getIndex()
{
    return m_index ; 
}  
