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
    virtual ~AssimpGGeo();
    int convert(const char* ctrl);
private:
    void init();
public:
    bool getVolNames();
public:
    static int load(GGeo* ggeo, const char* path, const char* query, const char* ctrl);

public:
    static const char* g4dae_bordersurface_physvolume1 ; 
    static const char* g4dae_bordersurface_physvolume2 ; 
    static const char* g4dae_skinsurface_volume ; 
    static const char* g4dae_opticalsurface_name ;
    static const char* g4dae_opticalsurface_type ;
    static const char* g4dae_opticalsurface_model ;
    static const char* g4dae_opticalsurface_finish ;
    static const char* g4dae_opticalsurface_value ;

protected:
    void convertMaterials(const aiScene* scene, GGeo* gg, const char* ctrl, bool reverse );
    void addProperties(GPropertyMap<float>* pmap, aiMaterial* material, bool reverse);
    void addPropertyVector(GPropertyMap<float>* pmap, const char* k, aiMaterialProperty* property, bool reverse);
    const char* getStringProperty(aiMaterial* mat, const char* query);

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
   m_volnames(false)
{
    init();
}




inline bool AssimpGGeo::getVolNames()
{
    return m_volnames ; 
}
