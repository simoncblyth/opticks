#include "CResource.hh"

CResource::CResource(unsigned int buffer_id, Access_t access ) 
    :
    m_imp(NULL),
    m_buffer_id(buffer_id),
    m_access(access),
    m_mapped(false)
{
    init();
}


