#pragma once

#include <map>
#include <vector>

#include <glm/glm.hpp>
#include "GVector.hh"
#include "GDomain.hh"
#include "GPropertyMap.hh"

class GMesh ; 
class GSolid ; 
class GNode ; 
class GMaterial ; 
class GSkinSurface ; 
class GBorderSurface ; 
class GBoundary ;
class GBoundaryLib ;
class GMergedMesh ;
class GSensorList ; 
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
        typedef std::map<unsigned int, std::string> Index_t ;
        static const char* GMERGEDMESH ; 
        static GGeo* load(const char* idpath, const char* mesh_version=NULL);
        enum { MAX_MERGED_MESH = 10 } ;
    private:
        void loadFromCache(const char* idpath);
    public:
        GGeo(bool loaded=false, bool volnames=false);  // loaded instances are only partial revivals

      
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
        void dumpStats(const char* msg="GGeo::dumpStats");
        unsigned int getNumMergedMesh();
        GMergedMesh* getMergedMesh(unsigned int index);
    public:
        // these are operational from cache
        // target 0 : all geometry of the mesh, >0 : specific volumes
        glm::vec4 getCenterExtent(unsigned int target, unsigned int merged_mesh_index=0u );
        void dumpTree(const char* msg="GGeo::dumpTree");  
        void dumpVolume(unsigned int index, const char* msg="GGeo::dumpVolume");  

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

    public:
        bool isLoaded();
        void setPath(const char* path);
        void setQuery(const char* query);
        void setCtrl(const char* ctrl);
        //void setVolNames(bool volnames);
        void setIdentityPath(const char* idpath);
        void setColors(GColors* colors);
        void setMeshVersion(const char* mesh_version);
        const char* getMeshVersion();
    public:
        char* getPath(); 
        char* getQuery(); 
        char* getCtrl(); 
        char* getIdentityPath(); 
        GColors* getColors();

    public:
        void add(GMesh*    mesh);
        void add(GSolid*    solid);

    public:
        void add(GMaterial* material);
        void add(GSkinSurface*  surface);
        void add(GBorderSurface*  surface);
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
        void sensitize(const char* idpath, const char* ext="idmap");
    private:
        void sensitize_traverse(GNode* node, unsigned int depth);
 
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
        GBoundaryLib* getBoundaryLib();
        GSensorList*  getSensorList();

    public:
        GMesh* getMesh(unsigned int index);  
        GMaterial* getMaterial(unsigned int index);  
        GSkinSurface* getSkinSurface(unsigned int index);  
        GBorderSurface* getBorderSurface(unsigned int index);  

    public:
        void targetTorchStep(TorchStepNPY* torchstep);

    public:
        std::vector<GMaterial*> getRawMaterialsWithProperties(const char* props, const char* delim);
        void findScintillators(const char* props);
        void dumpScintillators(const char* msg="GGeo::dumpScintillators");
        unsigned int getNumScintillators();
        GMaterial* getScintillator(unsigned int index);

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
        GSkinSurface* findSkinSurface(const char* lv);  
        GBorderSurface* findBorderSurface(const char* pv1, const char* pv2);  

    public:
        GMergedMesh* makeMergedMesh(unsigned int index=0, GNode* base=NULL);

    public:
        std::map<unsigned int, unsigned int>& getMeshUsage();
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

    private:
        bool                          m_loaded ;  
        std::vector<GMesh*>           m_meshes ; 
        std::vector<GSolid*>          m_solids ; 
        std::vector<GMaterial*>       m_materials ; 
        std::vector<GSkinSurface*>    m_skin_surfaces ; 
        std::vector<GBorderSurface*>  m_border_surfaces ; 

        // _raw mainly for debug
        std::vector<GMaterial*>       m_materials_raw ; 
        std::vector<GSkinSurface*>    m_skin_surfaces_raw ; 
        std::vector<GBorderSurface*>  m_border_surfaces_raw ; 
        std::vector<GMaterial*>       m_scintillators_raw ; 

        GBoundaryLib*                 m_boundary_lib ; 
        GSensorList*                  m_sensor_list ; 
        gfloat3*                      m_low ; 
        gfloat3*                      m_high ; 
        std::map<unsigned int,GMergedMesh*>     m_merged_mesh ; 
        std::map<unsigned int, unsigned int>    m_mesh_usage ; 
        GColors*                      m_colors ; 
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
        unsigned int                       m_sensitize_count ;  
        bool                               m_volnames ;    

};


inline GGeo::GGeo(bool loaded, bool volnames) :
   m_loaded(loaded), 
   m_boundary_lib(NULL),
   m_sensor_list(NULL),
   m_low(NULL),
   m_high(NULL),
   m_colors(NULL),
   m_meshindex(NULL),
   m_pvlist(NULL),
   m_lvlist(NULL),
   m_path(NULL),
   m_query(NULL),
   m_ctrl(NULL),
   m_idpath(NULL),
   m_mesh_version(NULL),
   m_sensitize_count(0),
   m_volnames(volnames)
{
   init(); 
}


inline bool GGeo::isLoaded()
{
    return m_loaded ; 
}


//inline void GGeo::setVolNames(bool volnames)
//{
//    m_volnames = volnames ; 
//}

inline void GGeo::add(GMaterial* material)
{
    m_materials.push_back(material);
    addToIndex((GPropertyMap<float>*)material);
}
inline void GGeo::add(GBorderSurface* surface)
{
    m_border_surfaces.push_back(surface);
    addToIndex((GPropertyMap<float>*)surface);
}
inline void GGeo::add(GSkinSurface* surface)
{
    m_skin_surfaces.push_back(surface);
    addToIndex((GPropertyMap<float>*)surface);
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
    return m_boundary_lib ; 
}
inline GSensorList* GGeo::getSensorList()
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

inline void GGeo::setColors(GColors* colors)
{
   m_colors = colors ; 
}
inline GColors* GGeo::getColors()
{
   return m_colors ; 
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





