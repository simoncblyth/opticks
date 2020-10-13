#include "PLOG.hh"
#include "NPY.hpp"
#include "NPoint.hpp"

const plog::Severity NPoint::LEVEL = PLOG::EnvLevel("NPoint","INFO"); 


NPoint::NPoint(unsigned ni)
    :
    m_arr(NPY<float>::make(ni,4))
{
    init(); 
}

void NPoint::init()
{
    m_arr->zero(); 
}


unsigned NPoint::getNum() const 
{
    return m_arr->getShape(0); 
}

void NPoint::add(const glm::vec3& v, float w )
{
    glm::vec4 q(v.x,v.y,v.z, w); 
    m_arr->add(q); 
}

void NPoint::add(const glm::vec4& q )
{
    m_arr->add(q); 
}


void NPoint::set(unsigned i, const glm::vec3& v, float w) 
{
    glm::vec4 q(v.x,v.y,v.z, w); 
    set(i, q);     
}
void NPoint::set(unsigned i, float x, float y, float z, float w) 
{
    glm::vec4 q(x,y,z,w); 
    set(i, q);     
}
void NPoint::set(unsigned i, const glm::vec4& q) const 
{
    assert( i < m_arr->getShape(0) ); 
    m_arr->setQuad(q, i, 0,0) ; 
}


glm::vec4 NPoint::get(unsigned i) const 
{
    return m_arr->getQuad_(i, 0,0) ; 
}




bool NPoint::HasSameDigest(const NPoint* a , const NPoint* b)
{
    std::string da = a->digest(); 
    std::string db = b->digest(); 
    return strcmp(da.c_str(), db.c_str()) == 0 ; 
}

NPoint* NPoint::MakeTransformed( const NPoint* src, const glm::mat4& transform ) // static
{
    unsigned n = src->getNum(); 
    NPoint* dst = new NPoint(n) ; 
    for(unsigned i=0 ; i < n ; i++) 
    {
        glm::vec4 a = src->get(i); 
        glm::vec4 b = transform * a ;    
        dst->set(i, b); 
    }
    return dst ;  
}

NPoint* NPoint::spawnTransformed( const glm::mat4& transform )
{
    return MakeTransformed(this, transform); 
}

std::string NPoint::digest() const 
{
    return m_arr->getDigestString(); 
}

std::string NPoint::desc(unsigned i) const 
{
    glm::vec4 q = get(i); 
    return glm::to_string(q);  
}

void NPoint::dump(const char* msg) const 
{
    unsigned n = getNum(); 
    LOG(info) << msg << " Num :" << n << " digest:" << digest() ; 
    for(unsigned i=0 ; i < n ; i++)
    {
        std::cout 
            << std::setw(7) << i 
            << " : "        
            << desc(i) 
            << std::endl
            ; 
    }
}

