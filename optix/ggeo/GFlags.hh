#pragma once

#include <cstddef>

class GCache ; 
class GAttrSeq ; 
class Index ; 

// replacing part of Types

class GFlags {
    public:
        GFlags(GCache* cache, const char* path="$ENV_HOME/graphics/optixrap/cu/photon.h");
    private:
        void init(const char* path);
        Index* parseFlags(const char* path);
    public:
        Index*      getIndex();  
        GAttrSeq*   getAttrIndex();  
    private:
        GCache*      m_cache  ;
        GAttrSeq*    m_aindex ; 
        Index*       m_index ; 
};


inline GFlags::GFlags(GCache* cache, const char* path) 
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

