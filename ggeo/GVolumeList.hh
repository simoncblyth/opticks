#pragma once

#include <vector>

class GVolume ; 

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

/**
GVolumeList
===========

Simple list of GVolume instances, used by GGeoTest.

**/


class GGEO_API GVolumeList 
{  
    public:
        GVolumeList();
        void add(GVolume* volume);
        unsigned getNumVolumes();
        GVolume* getVolume(unsigned index);
        std::vector<GVolume*>& getList();
    private:
        std::vector<GVolume*> m_volumes ; 

};

#include "GGEO_TAIL.hh"


