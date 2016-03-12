#pragma once

#include <vector>
#include "G4Transform3D.hh"
#include "G4MaterialPropertyVector.hh"

class G4VPhysicalVolume ;
class G4LogicalVolume ;
class G4Material ; 


class CTraverser {
    public:
        static const char* GROUPVEL ; 
    public:
        CTraverser(G4VPhysicalVolume* top);
    public:
        void Traverse();
        void createGroupVel();
        void dumpMaterials(const char* msg="CTraverser::dumpMaterials");
        void setVerbosity(unsigned int verbosity);
    private:
        G4Transform3D TraverseVolumeTree(const G4LogicalVolume* const volumePtr, const G4int depth);
        void Visit(const G4LogicalVolume* const lv);
        bool hasMaterial(const G4Material* material) ; 
        void addMaterial(const G4Material* material) ; 
        void dumpMaterial(const G4Material* material);
        void dumpMaterialProperty(const G4String& name, const G4MaterialPropertyVector* pvec);
    private:
        G4VPhysicalVolume* m_top ; 
        std::vector<const G4Material*> m_materials; ;
        unsigned int   m_verbosity ; 

};


inline CTraverser::CTraverser(G4VPhysicalVolume* top) 
   :
   m_top(top),
   m_verbosity(1)
{
}


inline void CTraverser::setVerbosity(unsigned int verbosity)
{
    m_verbosity = verbosity ; 
}
