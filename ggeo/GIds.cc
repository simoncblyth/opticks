#include "NGLM.hpp"
#include "NPY.hpp"

#include "GIds.hh"


GIds::GIds(NPY<unsigned int>* buf) 
    :
     m_buffer(buf)
{
}

NPY<unsigned int>* GIds::getBuffer()
{
    return m_buffer ; 
}


GIds* GIds::make(unsigned int n)
{
    GIds* ids = new GIds();
    for(unsigned int i=0 ; i < n ; i++) ids->add(0,0,0,0);
    return ids ;
}

GIds* GIds::load(const char* path)
{
    NPY<unsigned int>* buf = NPY<unsigned int>::load(path);
    GIds* t = new GIds(buf) ;
    return t ; 
}

void GIds::save(const char* path)
{
    if(!m_buffer) return ; 
    m_buffer->save(path);
}

void GIds::add(const glm::uvec4& v)
{
    add(v.x, v.y, v.z, v.w);
}

void GIds::add(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
{
    if(m_buffer == NULL) m_buffer = NPY<unsigned int>::make(0, 4);
    m_buffer->add(x, y, z, w);
}

glm::uvec4 GIds::get(unsigned int i)
{
    assert( m_buffer && i < m_buffer->getNumItems() && m_buffer->getNumValues(1) == 4);
    glm::uvec4 v = glm::make_vec4(m_buffer->getValues() + i*4) ;
    return v ; 
}



