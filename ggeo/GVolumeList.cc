#include "GVolumeList.hh"


GVolumeList::GVolumeList()
{
}

unsigned GVolumeList::getNumVolumes()
{
    return m_volumes.size();
}

GVolume* GVolumeList::getVolume(unsigned index)
{
    return m_volumes[index] ; 
}

void GVolumeList::add( GVolume* volume )
{
    m_volumes.push_back(volume);
}

std::vector<GVolume*>& GVolumeList::getList()
{
    return m_volumes ; 
}

