#include "InstanceCuller.hh"


InstanceCuller::InstanceCuller(const char* tag, const char* dir, const char* incl_path)
    :
    RendererBase(tag, dir, incl_path),
    m_composition(NULL)
{
}


InstanceCuller::~InstanceCuller()
{
}

void InstanceCuller::setComposition(Composition* composition)
{
    m_composition = composition ;
}
Composition* InstanceCuller::getComposition()
{
    return m_composition ;
}
 
