#pragma once

#include "GPropertyMap.hh" 

class AssimpTree ; 
class AssimpSelection ; 
class AssimpNode ; 

class GGeo ; 
class GSurface ;
class GMaterial ;
class GSolid ; 


struct aiMaterial ;
struct aiMaterialProperty ;
struct aiScene ;

class AssimpGGeo {
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

protected:
    void convertMaterials(const aiScene* scene, GGeo* gg, const char* ctrl );
    void addProperties(GPropertyMap<float>* pmap, aiMaterial* material);
    void addPropertyVector(GPropertyMap<float>* pmap, const char* k, aiMaterialProperty* property);
    const char* getStringProperty(aiMaterial* mat, const char* query);
protected:
    bool hasVectorProperty(aiMaterial* material, const char* propname);
    aiMaterialProperty* getVectorProperty(aiMaterial* material, const char* propname );
protected:
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



inline AssimpGGeo::AssimpGGeo(GGeo* ggeo, AssimpTree* tree, AssimpSelection* selection) 
   : 
   m_ggeo(ggeo),
   m_tree(tree),
   m_selection(selection),
   m_domain_scale(1.f),
   m_values_scale(1.f),
   m_domain_reciprocal(true),
   m_skin_surface(0),
   m_inborder_surface(0),
   m_outborder_surface(0),
   m_no_surface(0),
   m_volnames(false),
   m_reverse(true),        // true: ascending wavelength ordering of properties
   m_cathode(NULL),
   m_verbosity(0)
{
    init();
}


inline bool AssimpGGeo::getVolNames()
{
    return m_volnames ; 
}

inline void AssimpGGeo::setVerbosity(unsigned int verbosity)
{
    m_verbosity = verbosity ; 
}

