#include "Device.hh"

Device::Device()
{
}

void Device::add(void* smth)
{
    m_uploads.push_back(smth);
}


bool Device::isUploaded(void* smth)
{
    for(unsigned int i=0 ; i < m_uploads.size() ; i++) if(m_uploads[i] == smth) return true ;   
    return false ; 
}

