#pragma once

class AssimpTree ; 
class AssimpSelection ; 
class AssimpNode ; 

class GGeo ; 
class GSurface ;
class GMaterial ;
class GSolid ; 
class GMesh ; 

template <typename T> class GPropertyMap ; 

struct aiMaterial ;
struct aiMaterialProperty ;
struct aiScene ;
struct aiMesh ;

#include "ASIRAP_API_EXPORT.hh"

class ASIRAP_API AssimpGGeo {
public:
    AssimpGGeo(GGeo* ggeo, AssimpTree* tree, AssimpSelection* selection);
    int convert(const char* ctrl);
private:
    void init();
public:
    bool getVolNames();
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
    void convertStructure(GGeo* gg, AssimpNode* node, unsigned int depth, GSolid* parent);
    GSolid* convertStructureVisit(GGeo* gg, AssimpNode* node, unsigned int depth, GSolid* parent);

protected:
    void setDomainScale(float dscale);
    void setValuesScale(float vscale);

private:
    void convertSensors(GGeo* gg);
    void convertSensors(GGeo* gg, AssimpNode* node, unsigned int depth);
    void convertSensorsVisit(GGeo* gg, AssimpNode* node, unsigned int depth);

private:
    GGeo*            m_ggeo ;
    AssimpTree*      m_tree ; 
    AssimpSelection* m_selection ;
 
    float            m_domain_scale ; 
    float            m_values_scale ; 
    bool             m_domain_reciprocal ; 

    unsigned int     m_skin_surface ; 
    unsigned int     m_inborder_surface ; 
    unsigned int     m_outborder_surface ; 
    unsigned int     m_no_surface ; 

    bool             m_volnames ; 
    bool             m_reverse ; 
    aiMaterial*      m_cathode ; 
    float            m_fake_efficiency ; 

    unsigned int     m_verbosity ; 

};



