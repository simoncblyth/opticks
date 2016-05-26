#pragma once

#include <glm/glm.hpp>
#include <vector>
#include "G4Transform3D.hh"
#include "G4MaterialPropertyVector.hh"

template <typename T> class NPY ;

class G4VPhysicalVolume ;
class G4LogicalVolume ;
class G4Material ; 


class CTraverser {
    public:
        static const char* GROUPVEL ; 
    public:
        CTraverser(G4VPhysicalVolume* top);
    private:
        void init();
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
    public:
        const char* getPVName(unsigned int index);
        glm::mat4 getGlobalTransform(unsigned int index);
        glm::mat4 getLocalTransform(unsigned int index);
        unsigned int getNumGlobalTransforms();
        unsigned int getNumLocalTransforms();
        NPY<float>*  getGlobalTransforms();
        NPY<float>*  getLocalTransforms();
    private:
        void collectTransformT(NPY<float>* buffer, const G4Transform3D& T);
        void collectTransform(NPY<float>* buffer, const G4Transform3D& T);
        void AncestorVisit(std::vector<const G4VPhysicalVolume*> ancestors);
        void AncestorTraverse(std::vector<const G4VPhysicalVolume*> ancestors, const G4VPhysicalVolume* pv);
    private:
        G4Transform3D TraverseVolumeTree(const G4LogicalVolume* const volumePtr, const G4int depth);
        void Visit(const G4LogicalVolume* const lv);
        void VisitPV(const G4VPhysicalVolume* const pv, const G4Transform3D& T );

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
        unsigned int   m_gcount ; 
        unsigned int   m_lcount ; 
        NPY<float>*    m_gtransforms ; 
        NPY<float>*    m_ltransforms ; 
        std::vector<std::string> m_pvnames ; 


};



inline CTraverser::CTraverser(G4VPhysicalVolume* top) 
   :
   m_top(top),
   m_verbosity(1),
   m_gcount(0),
   m_lcount(0),
   m_gtransforms(NULL),
   m_ltransforms(NULL)
{
   init();
}



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
inline void CTraverser::setVerbosity(unsigned int verbosity)
{
    m_verbosity = verbosity ; 
}

inline NPY<float>* CTraverser::getGlobalTransforms()
{
    return m_gtransforms ; 
}
inline NPY<float>* CTraverser::getLocalTransforms()
{
    return m_ltransforms ; 
}




