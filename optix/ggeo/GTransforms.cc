#include "NGLM.hpp"
#include "NPY.hpp"

#include "GTransforms.hh"


GTransforms::GTransforms(NPY<float>* buf) 
    :
    m_buffer(buf)
{
}

NPY<float>* GTransforms::getBuffer()
{
    return m_buffer ; 
}


GTransforms* GTransforms::make(unsigned int n)
{
    GTransforms* t = new GTransforms();
    for(unsigned int i=0 ; i < n ; i++) t->add();
    return t ;
}

GTransforms* GTransforms::load(const char* path)
{
    NPY<float>* buf = NPY<float>::load(path);
    GTransforms* t = new GTransforms(buf) ;
    return t ; 
}

void GTransforms::save(const char* path)
{
    if(!m_buffer) return ; 
    m_buffer->save(path);
}


void GTransforms::add(const glm::mat4& mat)
{
    if(m_buffer == NULL) m_buffer = NPY<float>::make(0, 4, 4);
    m_buffer->add( glm::value_ptr(mat), 4*4 );
}

void GTransforms::add()
{
    glm::mat4 identity ; 
    add(identity);
}

glm::mat4 GTransforms::get(unsigned int i)
{
    assert( m_buffer && i < m_buffer->getNumItems() && m_buffer->getNumValues(1) == 16);
    glm::mat4 mat = glm::make_mat4(m_buffer->getValues() + i*16) ;
    return mat ; 
}



