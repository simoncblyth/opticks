#pragma once

#include <string>

class G4Material ; 

#include "CFG4_API_EXPORT.hh"
class CFG4_API CMaterial
{
    public: 
         static std::string Digest(G4Material* material);
    public:
         CMaterial(G4Material* mat) ; 
         G4Material* getMaterial() const ;  
    public:
         std::string  digest() const ;
    private:
         G4Material* m_material ; 

};
