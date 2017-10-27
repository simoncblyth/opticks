#pragma once

#include <vector>

class GSolid ; 

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GSolidList 
{  
    public:
        GSolidList();
        void add(GSolid* solid);
        unsigned getNumSolids();
        GSolid* getSolid(unsigned index);
        std::vector<GSolid*>& getList();
    private:
        std::vector<GSolid*> m_solids ; 

};

#include "GGEO_TAIL.hh"


