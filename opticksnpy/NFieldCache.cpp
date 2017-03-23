#include "NFieldCache.hpp"

NFieldCache::NFieldCache( std::function<float(float,float,float)> field, const nbbox& bb) 
    : 
     m_field(field), 
     m_bbox(bb),
     m_side({bb.max.x - bb.min.x, bb.max.y - bb.min.y, bb.max.z - bb.min.z})
{
     reset();
}

// https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
//
// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
unsigned expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

unsigned MortonCode( float sx, float sy, float sz)
{
    // 30-bit Morton code,  (sx,sy,sz) must be in 0. to 1023.
    // 1 << 10 = 1024 = 2^10

    unsigned xx = expandBits((unsigned int)sx);
    unsigned yy = expandBits((unsigned int)sy);
    unsigned zz = expandBits((unsigned int)sz);

    return (xx << 2) + (yy << 1) + zz;
}

unsigned NFieldCache::getMortonCode( float x, float y, float z)
{
    float sx = 1024.0f*(x - m_bbox.min.x)/m_side.x ; 
    float sy = 1024.0f*(y - m_bbox.min.y)/m_side.y ; 
    float sz = 1024.0f*(z - m_bbox.min.z)/m_side.z ; 
    return MortonCode(sx,sy,sz);
}


float NFieldCache::operator()(float x, float y, float z)
{
    unsigned morton = getMortonCode(x,y,z) ;
    float result = 0.f ; 

    UMAP::const_iterator it = m_cache.find(morton);
    if(it == m_cache.end())
    {
        result = m_field(x, y, z) ;
        m_cache.emplace(morton, result)  ;
        m_calc += 1 ; 
    }
    else
    {
        result = it->second ;     
        m_lookup += 1 ; 
    }
    return result ;
}


void NFieldCache::reset()
{
    m_lookup = 0 ; 
    m_calc  = 0 ; 
}

std::string NFieldCache::desc()
{
    std::stringstream ss ; 
    ss << " calc " << m_calc << " lookup " << m_lookup ; 
    return ss.str();
}


