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
        void setVerbosity(unsigned int verbosity);
    public:
        void dumpMaterials(const char* msg="CTraverser::dumpMaterials");
        unsigned int getNumMaterials();
        unsigned int getNumMaterialsWithoutMPT();
        const G4Material* getMaterial(unsigned int index);
        G4Material* getMaterialWithoutMPT(unsigned int index);
        void Summary(const char* msg="CTraverser::Summary"); 
    private:
        G4Transform3D TraverseVolumeTree(const G4LogicalVolume* const volumePtr, const G4int depth);
        void Visit(const G4LogicalVolume* const lv);
        bool hasMaterial(const G4Material* material) ; 
        void addMaterial(const G4Material* material) ; 
        void dumpMaterial(const G4Material* material);
        void dumpMaterialProperty(const G4String& name, const G4MaterialPropertyVector* pvec);
    private:
        bool hasMaterialWithoutMPT(G4Material* material) ; 
        void addMaterialWithoutMPT(G4Material* material) ; 
    private:
        G4VPhysicalVolume* m_top ; 
        std::vector<const G4Material*> m_materials ;
        std::vector<G4Material*>       m_materials_without_mpt ;
        unsigned int   m_verbosity ; 

};




inline unsigned int CTraverser::getNumMaterials()
{
   return m_materials.size();
}
inline unsigned int CTraverser::getNumMaterialsWithoutMPT()
{
   return m_materials_without_mpt.size();
}
inline const G4Material* CTraverser::getMaterial(unsigned int index)
{
   return m_materials[index];
}
inline G4Material* CTraverser::getMaterialWithoutMPT(unsigned int index)
{
   return m_materials_without_mpt[index];
}






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
