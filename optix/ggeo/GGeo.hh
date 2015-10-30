#pragma once

#include <map>
#include <vector>
#include <unordered_set>
#include <iterator>

#include <glm/glm.hpp>
#include "GVector.hh"
#include "GDomain.hh"
#include "GPropertyMap.hh"

class GCache; 
class GMesh ; 
class GSolid ; 
class GNode ; 
class GMaterial ; 
class GSkinSurface ; 
class GBorderSurface ; 
class GBoundary ;

class GBndLib ;
class GBoundaryLib ;
class GMaterialLib ;
class GSurfaceLib ;
class GScintillatorLib ;

class GTreeCheck ;
class GColorizer ; 
class GColors ; 

class GMergedMesh ;
class NSensorList ; 
class GColors ; 
class GItemIndex ; 
class GItemList ; 


class TorchStepNPY ; 
//
// NB GGeo is a dumb substrate from which the geometry model is created,
//    eg by AssimpGGeo::convert 
//
class GGeo {
    public:
        static const char* CATHODE_MATERIAL ; 
    public:
        typedef int (*GLoaderImpFunctionPtr)(GGeo*);
        void setLoaderImp(GLoaderImpFunctionPtr imp);
    public:
        typedef GMesh* (*GJoinImpFunctionPtr)(GMesh*, GCache*);
        void setMeshJoinImp(GJoinImpFunctionPtr imp);
        void setMeshJoinCfg(const char* config);
        bool shouldMeshJoin(GMesh* mesh);
        GMesh* invokeMeshJoin(GMesh* mesh);    // used from AssimpGGeo::convertMeshes immediately after GMesh birth and deduping
    public:
        typedef std::map<unsigned int, std::string> Index_t ;
        static const char* GMERGEDMESH ; 
        static GGeo* load(const char* idpath, const char* mesh_version=NULL);
        static bool ctrlHasKey(const char* ctrl, const char* key);
        enum { MAX_MERGED_MESH = 10 } ;
    public:
        GGeo(GCache* cache); 
        GCache* getCache();
        const char* getIdPath();
        void loadFromCache();
    public:
        void loadFromG4DAE();  // AssimpGGeo::load
        void prepareScintillatorLib();
        void prepareMeshes();
        void prepareVertexColors();
    public:
        const char* getPVName(unsigned int index);
        const char* getLVName(unsigned int index);

        void setCathode(GMaterial* cathode);
        GMaterial* getCathode();  

        virtual ~GGeo();
   private:
        void init(); 
        void loadMergedMeshes(const char* idpath);
        void removeMergedMeshes(const char* idpath);
    public:
        void save(const char* idpath);
    private:
        void saveMergedMeshes(const char* idpath);
    public:
        unsigned int getNumMergedMesh();
        GMergedMesh* getMergedMesh(unsigned int index);
    public:
        // these are operational from cache
        // target 0 : all geometry of the mesh, >0 : specific volumes
        glm::vec4 getCenterExtent(unsigned int target, unsigned int merged_mesh_index=0u );
        void dumpTree(const char* msg="GGeo::dumpTree");  
        void dumpVolume(unsigned int index, const char* msg="GGeo::dumpVolume");  
        void dumpNodeInfo(unsigned int mmindex, const char* msg="GGeo::dumpNodeInfo" );
        void dumpStats(const char* msg="GGeo::dumpStats");

        // merged mesh buffer offsets and counts
        //
        //     .x  prior faces offset    
        //     .y  prior verts offset  
        //     .z  index faces count
        //     .w  index verts count
        //
        glm::ivec4 getNodeOffsetCount(unsigned int index);
        glm::vec4 getFaceCenterExtent(unsigned int face_index, unsigned int solid_index, unsigned int mergedmesh_index=0 );
        glm::vec4 getFaceRangeCenterExtent(unsigned int face_index0, unsigned int face_index1, unsigned int solid_index, unsigned int mergedmesh_index=0 );
        glm::mat4 getTransform(unsigned int index);
    public:
        bool isLoaded();
        bool isVolnames();

        void setPath(const char* path);
        void setQuery(const char* query);
        void setCtrl(const char* ctrl);
        //void setVolNames(bool volnames);
        void setIdentityPath(const char* idpath);
        void setMeshVersion(const char* mesh_version);
        const char* getMeshVersion();
    public:
        char* getPath(); 
        char* getQuery(); 
        char* getCtrl(); 
        char* getIdentityPath(); 

    public:
        void add(GMesh*    mesh);
        void add(GSolid*    solid);

    public:
        void add(GMaterial* material);
        void add(GSkinSurface*  surface);
        void add(GBorderSurface*  surface);

        void close();

        void addToIndex(GPropertyMap<float>* obj);
        void dumpIndex(const char* msg="GGeo::dumpIndex");

    public:
        void addRaw(GMaterial* material);
        void addRaw(GSkinSurface* surface);
        void addRaw(GBorderSurface*  surface);
      
    public:
        GItemIndex*  getMeshIndex(); 
        GItemList*   getPVList(); 
        GItemList*   getLVList(); 

    public:
        void dumpRaw(const char* msg="GGeo::dumpRaw");
        void dumpRawMaterialProperties(const char* msg="GGeo::dumpRawMaterialProperties");
        void dumpRawSkinSurface(const char* name=NULL);
        void dumpRawBorderSurface(const char* name=NULL);

    public:
        // load idmap, traverse GNode tree calling GSolid::setSensor nodes with associated sensor identifier
        NSensorList*  getSensorList();

    public:
        void traverse(const char* msg="GGeo::traverse");
    private:
        void traverse(GNode* node, unsigned int depth);
 
    public:
        unsigned int getNumMeshes();
        unsigned int getNumSolids();
        unsigned int getNumMaterials();
        unsigned int getNumSkinSurfaces();
        unsigned int getNumBorderSurfaces();
    public:
        unsigned int getNumRawMaterials();
        unsigned int getNumRawSkinSurfaces();
        unsigned int getNumRawBorderSurfaces();

    public:
        GBndLib*      getBndLib();
        GBoundaryLib* getBoundaryLib();
        GMaterialLib* getMaterialLib();
        GSurfaceLib*  getSurfaceLib();
        GScintillatorLib*  getScintillatorLib();
        GColorizer*        getColorizer();
        GColors*           getColors();

    public:
        GMesh* getMesh(unsigned int index);  
        GMaterial* getMaterial(unsigned int index);  
        GSkinSurface* getSkinSurface(unsigned int index);  
        GBorderSurface* getBorderSurface(unsigned int index);  

    public:
        void targetTorchStep(TorchStepNPY* torchstep);

    public:
        void findScintillatorMaterials(const char* props);
        void dumpScintillatorMaterials(const char* msg="GGeo::dumpScintillatorMaterials");
        unsigned int getNumScintillatorMaterials();
        GMaterial* getScintillatorMaterial(unsigned int index);
    public:
        void findCathodeMaterials(const char* props);
        void dumpCathodeMaterials(const char* msg="GGeo::dumpCathodeMaterials");
        unsigned int getNumCathodeMaterials();
        GMaterial* getCathodeMaterial(unsigned int index);
    public:
        std::vector<GMaterial*> getRawMaterialsWithProperties(const char* props, const char* delim);
    public:
        GPropertyMap<float>* findRawMaterial(const char* shortname);
        GProperty<float>*    findRawMaterialProperty(const char* shortname, const char* propname);

    public:
        GSolid* getSolid(unsigned int index);  
        GSolid* getSolidSimple(unsigned int index);  

    public:
        gfloat3* getLow();
        gfloat3* getHigh();
        void setLow(const gfloat3& low);
        void setHigh(const gfloat3& high);
        void updateBounds(GNode* node); 

    public:
        void addCathodeLV(const char* lv);
        void dumpCathodeLV(const char* msg="GGeo::dumpCathodeLV");
        const char* getCathodeLV(unsigned int index);
        unsigned int getNumCathodeLV();
    public:
        GSkinSurface* findSkinSurface(const char* lv);  
        GBorderSurface* findBorderSurface(const char* pv1, const char* pv2);  

    public:
        GMergedMesh* makeMergedMesh(unsigned int index=0, GNode* base=NULL);

    public:
        std::map<unsigned int, unsigned int>& getMeshUsage();
        std::map<unsigned int, std::vector<unsigned int> >& getMeshNodes();
        void countMeshUsage(unsigned int meshIndex, unsigned int nodeIndex, const char* lv, const char* pv);
        void reportMeshUsage(const char* msg="GGeo::reportMeshUsage");

#if 0
    public:
        void materialConsistencyCheck();
        unsigned int materialConsistencyCheck(GSolid* solid);
#endif

    public:
        void Summary(const char* msg="GGeo::Summary");
        void Details(const char* msg="GGeo::Details");

    public:
        GTreeCheck* getTreeCheck();
    private:
        GCache*                       m_cache ; 
        GTreeCheck*                   m_treecheck ; 
        bool                          m_loaded ;  
        std::vector<GMesh*>           m_meshes ; 
        std::vector<GSolid*>          m_solids ; 
        std::vector<GMaterial*>       m_materials ; 
        std::vector<GSkinSurface*>    m_skin_surfaces ; 
        std::vector<GBorderSurface*>  m_border_surfaces ; 

        std::vector<GSolid*>           m_sensitive_solids ; 
        std::unordered_set<GBoundary*> m_sensitive_boundaries ; 

        std::unordered_set<std::string> m_cathode_lv ; 

        // _raw mainly for debug
        std::vector<GMaterial*>       m_materials_raw ; 
        std::vector<GSkinSurface*>    m_skin_surfaces_raw ; 
        std::vector<GBorderSurface*>  m_border_surfaces_raw ; 

        std::vector<GMaterial*>       m_scintillators_raw ; 
        std::vector<GMaterial*>       m_cathodes_raw ; 

        GBndLib*                      m_bndlib ; 
        GBoundaryLib*                 m_boundarylib ; 
        GMaterialLib*                 m_materiallib ; 
        GSurfaceLib*                  m_surfacelib ; 
        GScintillatorLib*             m_scintillatorlib ; 
        GColorizer*                   m_colorizer ; 

        NSensorList*                  m_sensor_list ; 
        gfloat3*                      m_low ; 
        gfloat3*                      m_high ; 
        std::map<unsigned int,GMergedMesh*>     m_merged_mesh ; 
        std::map<unsigned int, unsigned int>    m_mesh_usage ; 
        std::map<unsigned int, std::vector<unsigned int> >    m_mesh_nodes ; 
        GItemIndex*                   m_meshindex ; 
        GItemList*                    m_pvlist ; 
        GItemList*                    m_lvlist ; 

        char*                         m_path ;
        char*                         m_query ;
        char*                         m_ctrl ;
        char*                         m_idpath ;
        char*                         m_mesh_version ;

    private:
        std::map<unsigned int, GSolid*>    m_solidmap ; 
        Index_t                            m_index ; 
        unsigned int                       m_sensitive_count ;  
        bool                               m_volnames ;    
        GMaterial*                         m_cathode ; 
        const char*                        m_join_cfg ; 
        GJoinImpFunctionPtr                m_join_imp ;  
        GLoaderImpFunctionPtr              m_loader_imp ;  

};


inline GGeo::GGeo(GCache* cache) :
   m_cache(cache), 
   m_treecheck(NULL), 
   m_loaded(false), 
   m_bndlib(NULL),
   m_boundarylib(NULL),
   m_materiallib(NULL),
   m_surfacelib(NULL),
   m_scintillatorlib(NULL),
   m_colorizer(NULL),
   m_sensor_list(NULL),
   m_low(NULL),
   m_high(NULL),
   m_meshindex(NULL),
   m_pvlist(NULL),
   m_lvlist(NULL),
   m_path(NULL),
   m_query(NULL),
   m_ctrl(NULL),
   m_idpath(NULL),
   m_mesh_version(NULL),
   m_sensitive_count(0),
   m_volnames(false),
   m_cathode(NULL),
   m_join_cfg(NULL)
{
   init(); 
}



// setLoaderImp : sets implementation that does the actual loading
// using a function pointer to the implementation 
// avoids ggeo-/GLoader depending on all the implementations

inline void GGeo::setLoaderImp(GLoaderImpFunctionPtr imp)
{
    m_loader_imp = imp ; 
}
inline void GGeo::setMeshJoinImp(GJoinImpFunctionPtr imp)
{
    m_join_imp = imp ; 
}
inline void GGeo::setMeshJoinCfg(const char* cfg)
{
    m_join_cfg = cfg ? strdup(cfg) : NULL  ; 
}

inline bool GGeo::isLoaded()
{
    return m_loaded ; 
}

inline bool GGeo::isVolnames()
{
    return m_volnames ; 
}


inline void GGeo::addRaw(GMaterial* material)
{
    m_materials_raw.push_back(material);
}
inline void GGeo::addRaw(GBorderSurface* surface)
{
    m_border_surfaces_raw.push_back(surface);
}
inline void GGeo::addRaw(GSkinSurface* surface)
{
    m_skin_surfaces_raw.push_back(surface);
}


inline unsigned int GGeo::getNumMeshes()
{
    return m_meshes.size();
}
inline unsigned int GGeo::getNumSolids()
{
    return m_solids.size();
}
inline unsigned int GGeo::getNumMaterials()
{
    return m_materials.size();
}
inline unsigned int GGeo::getNumBorderSurfaces()
{
    return m_border_surfaces.size();
}
inline unsigned int GGeo::getNumSkinSurfaces()
{
    return m_skin_surfaces.size();
}
inline unsigned int GGeo::getNumRawMaterials()
{
    return m_materials_raw.size();
}
inline unsigned int GGeo::getNumRawBorderSurfaces()
{
    return m_border_surfaces_raw.size();
}
inline unsigned int GGeo::getNumRawSkinSurfaces()
{
    return m_skin_surfaces_raw.size();
}




inline GSolid* GGeo::getSolidSimple(unsigned int index)
{
    return m_solids[index];
}
inline GSkinSurface* GGeo::getSkinSurface(unsigned int index)
{
    return m_skin_surfaces[index];
}
inline GBorderSurface* GGeo::getBorderSurface(unsigned int index)
{
    return m_border_surfaces[index];
}



inline char* GGeo::getPath()
{
   return m_path ;
}
inline char* GGeo::getQuery()
{
   return m_query ;
}
inline char* GGeo::getCtrl()
{
   return m_ctrl ;
}
inline char* GGeo::getIdentityPath()
{
   return m_idpath ;
}


inline GBoundaryLib* GGeo::getBoundaryLib()
{
    return m_boundarylib ; 
}
inline GBndLib* GGeo::getBndLib()
{
    return m_bndlib ; 
}
inline GMaterialLib* GGeo::getMaterialLib()
{
    return m_materiallib ; 
}
inline GSurfaceLib* GGeo::getSurfaceLib()
{
    return m_surfacelib ; 
}
inline GScintillatorLib* GGeo::getScintillatorLib()
{
    return m_scintillatorlib ; 
}
inline GColorizer* GGeo::getColorizer()
{
    return m_colorizer ; 
}






inline NSensorList* GGeo::getSensorList()
{
    return m_sensor_list ; 
}



inline gfloat3* GGeo::getLow()
{
   return m_low ; 
}
inline gfloat3* GGeo::getHigh()
{
   return m_high ; 
}


inline GItemIndex* GGeo::getMeshIndex()
{
    return m_meshindex ; 
}
inline GItemList* GGeo::getPVList()
{
    return m_pvlist ; 
}
inline GItemList* GGeo::getLVList()
{
    return m_lvlist ; 
}

inline void GGeo::setMeshVersion(const char* mesh_version)
{
    m_mesh_version = mesh_version ? strdup(mesh_version) : NULL ;
}




inline const char* GGeo::getMeshVersion()
{
    return m_mesh_version ;
}

inline GCache* GGeo::getCache()
{
    return m_cache ; 
}
inline GTreeCheck* GGeo::getTreeCheck()
{
    return m_treecheck ;
}

inline GMaterial* GGeo::getCathode()
{
    return m_cathode ; 
}
inline void GGeo::setCathode(GMaterial* cathode)
{
    m_cathode = cathode ; 
}

inline void GGeo::addCathodeLV(const char* lv)
{
   m_cathode_lv.insert(lv);
}

inline unsigned int GGeo::getNumCathodeLV()
{
   return m_cathode_lv.size() ; 
}
inline const char* GGeo::getCathodeLV(unsigned int index)
{
    typedef std::unordered_set<std::string>::const_iterator UCI ; 
    UCI it = m_cathode_lv.begin() ; 
    std::advance( it, index );
    return it != m_cathode_lv.end() ? it->c_str() : NULL  ; 
}

inline void GGeo::dumpCathodeLV(const char* msg)
{
    printf("%s\n", msg);
    typedef std::unordered_set<std::string>::const_iterator UCI ; 
    for(UCI it=m_cathode_lv.begin() ; it != m_cathode_lv.end() ; it++)
    {
        printf("GGeo::dumpCathodeLV %s \n", it->c_str() ); 
    }
}


