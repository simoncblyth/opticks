#pragma once

#include <vector>

class G4OpticalSurface ; 

#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API CSurfaceTable {
    public:
         CSurfaceTable(const char* name);
         unsigned getNumSurf();
         const char* getName();
         const G4OpticalSurface* getSurface(unsigned index);
    protected: 
         void add(const G4OpticalSurface* surf);
    private:
         const char* m_name ;
         std::vector<const G4OpticalSurface*>  m_surfaces ;  
};

#include "CFG4_TAIL.hh"

