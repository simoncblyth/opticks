#pragma once


class NSensorList ; 
class Opticks ; 


class AssimpTree ; 
class AssimpSelection ; 
class AssimpNode ; 

class GGeo ; 
class GSurface ;
class GMaterial ;
class GVolume ; 
class GMesh ; 
class OpticksQuery ; 

template <typename T> class GPropertyMap ; 

struct aiMaterial ;
struct aiMaterialProperty ;
struct aiScene ;
struct aiMesh ;

#include "ASIRAP_API_EXPORT.hh"

/*
AssimpGGeo
===========

Primary entry point is AssimpGGeo::load(GGeo* ggeo)
which imports the full geometry Assimp node tree.
The Assimp node tree is converted into the GVolume/GNode tree, 
with AssimpSelection such as volume ranges feeding into GVolume/GNode 
selected flags (GVolume::setSelected).

* think of the selection as a node mask.

::

     806 void AssimpGGeo::convertStructure(GGeo* gg, AssimpNode* node, unsigned int depth, GVolume* parent)
     807 {
     808     // recursive traversal of the AssimpNode tree
     809     // note that full tree is traversed even when a partial selection is applied 
     ...
     812     GVolume* solid = convertStructureVisit( gg, node, depth, parent);
     814     bool selected = m_selection && m_selection->contains(node) ;
     816     solid->setSelected(selected);
     817 
     818     gg->add(solid);
     ...
     830     for(unsigned int i = 0; i < node->getNumChildren(); i++) convertStructure(gg, node->getChild(i), depth + 1, solid);
     831 }

*/


class ASIRAP_API AssimpGGeo {
public:
    AssimpGGeo(GGeo* ggeo, AssimpTree* tree, AssimpSelection* selection, OpticksQuery* query);
    int convert(const char* ctrl);
private:
    void init();
public:
    //bool getVolNames();
    void setVerbosity(unsigned int verbosity);
public:
    static int load(GGeo* ggeo);

public:
    static const char* g4dae_bordersurface_physvolume1 ; 
    static const char* g4dae_bordersurface_physvolume2 ; 
    static const char* g4dae_skinsurface_volume ; 
    static const char* g4dae_opticalsurface_name ;
    static const char* g4dae_opticalsurface_type ;
    static const char* g4dae_opticalsurface_model ;
    static const char* g4dae_opticalsurface_finish ;
    static const char* g4dae_opticalsurface_value ;

    static const char* g4dae_material_srcidx ; 

    static const char* EFFICIENCY ;

public:
    // used for debugging
    GMesh* convertMesh(unsigned int index );
    GMesh* convertMesh(const char* name);  // name eg iav or oav 
    unsigned int getNumMeshes();
protected:
    void convertMaterials(const aiScene* scene, GGeo* gg, const char* ctrl );
    void addProperties(GPropertyMap<float>* pmap, aiMaterial* material);
    void addPropertyVector(GPropertyMap<float>* pmap, const char* k, aiMaterialProperty* property);
    const char* getStringProperty(aiMaterial* mat, const char* query);
protected:
    bool hasVectorProperty(aiMaterial* material, const char* propname);
    aiMaterialProperty* getVectorProperty(aiMaterial* material, const char* propname );
protected:
    GMesh* convertMesh(const aiMesh* mesh, unsigned int index );
    void convertMeshes(const aiScene* scene, GGeo* gg, const char* ctrl);
protected:
    void convertStructure(GGeo* gg);
    void convertStructure(GGeo* gg, AssimpNode* node, unsigned int depth, GVolume* parent);
    GVolume* convertStructureVisit(GGeo* gg, AssimpNode* node, unsigned int depth, GVolume* parent);

protected:
    void setDomainScale(float dscale);
    void setValuesScale(float vscale);

private:
    void convertSensors(GGeo* gg);
    void convertSensors(GGeo* gg, AssimpNode* node, unsigned int depth);
    void convertSensorsVisit(GGeo* gg, AssimpNode* node, unsigned int depth);

private:
    Opticks*         m_ok ; 
    NSensorList*     m_sensor_list ;  
    GGeo*            m_ggeo ;
    AssimpTree*      m_tree ; 
    AssimpSelection* m_selection ;
    OpticksQuery*    m_query ; 
    bool             m_nosel ;
 
    float            m_domain_scale ; 
    float            m_values_scale ; 
    bool             m_domain_reciprocal ; 

    unsigned int     m_skin_surface ; 
    unsigned int     m_inborder_surface ; 
    unsigned int     m_outborder_surface ; 
    unsigned int     m_no_surface ; 

    bool             m_volnames ; 
    bool             m_reverse ; 
    aiMaterial*      m_cathode_amat ; 
    GMaterial*       m_cathode_gmat ; 
    float            m_fake_efficiency ; 

    unsigned int     m_verbosity ; 

};



