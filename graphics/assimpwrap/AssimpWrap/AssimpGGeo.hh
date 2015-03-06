#ifndef ASSIMPGGEO_H
#define ASSIMPGGEO_H

class AssimpTree ; 
class AssimpNode ; 
class GGeo ; 
class GSurface ;
class GMaterial ;
class GPropertyMap ;
class GSolid ; 

struct aiMaterial ;
struct aiMaterialProperty ;
struct aiScene ;

class AssimpGGeo {
public:
    AssimpGGeo(AssimpTree* tree);
    virtual ~AssimpGGeo();

public:
    GGeo* convert(const char* ctrl);

public:
    static const char* g4dae_bordersurface_physvolume1 ; 
    static const char* g4dae_bordersurface_physvolume2 ; 
    static const char* g4dae_skinsurface_volume ; 

protected:
    void convertMaterials(const aiScene* scene, GGeo* gg, const char* ctrl);
    void addProperties(GPropertyMap* pmap, aiMaterial* material);
    void addPropertyVector(GPropertyMap* pmap, const char* k, aiMaterialProperty* property );
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
    AssimpTree* m_tree ; 
    float m_domain_scale ; 
    float m_values_scale ; 
    bool m_domain_reciprocal ; 

    unsigned int m_skin_surface ; 
    unsigned int m_inborder_surface ; 
    unsigned int m_outborder_surface ; 
    unsigned int m_no_surface ; 

};

#endif





