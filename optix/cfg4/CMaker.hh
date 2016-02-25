#pragma once

#include <glm/glm.hpp>
class GCache ; 
class GCSG ; 

class G4VSolid;

class CMaker {
    public:
        CMaker(GCache* cache);
    public:
        G4VSolid* makeSolid(char shapecode, const glm::vec4& param);
        G4VSolid* makeBox(const glm::vec4& param);
        G4VSolid* makeSphere(const glm::vec4& param);
    public:
        G4VSolid* makeSolid(GCSG* csg, unsigned int i);
    private:
        GCache* m_cache ; 

};

inline CMaker::CMaker(GCache* cache) 
   :
   m_cache(cache)
{
}   


