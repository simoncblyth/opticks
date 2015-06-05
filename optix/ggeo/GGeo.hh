#pragma once

#include <map>
#include <vector>

#include "GVector.hh"
#include "GDomain.hh"
#include "GPropertyMap.hh"

class GMesh ; 
class GSolid ; 
class GNode ; 
class GMaterial ; 
class GSkinSurface ; 
class GBorderSurface ; 
class GSubstance ;
class GSubstanceLib ;
class GMergedMesh ;

//
// NB GGeo is a dumb substrate from which the geometry model is created,
//    eg by AssimpGGeo::convert 
//
class GGeo {
    public:
        GGeo();
        virtual ~GGeo();
    private:
        void init(); 

    public:
        void setPath(const char* path);
        void setQuery(const char* query);
        void setCtrl(const char* ctrl);
        void setIdentityPath(const char* idpath);

    public:
        char* getPath(); 
        char* getQuery(); 
        char* getCtrl(); 
        char* getIdentityPath(); 

    public:
        void add(GMaterial* material);
        void add(GMesh*    mesh);
        void add(GSolid*    solid);
        void add(GSkinSurface*  surface);
        void add(GBorderSurface*  surface);
    public:
        void addRaw(GMaterial* material);
        void addRaw(GSkinSurface* surface);
        void addRaw(GBorderSurface*  surface);

    public:
        void dumpRaw(const char* msg="GGeo::dumpRaw");
        void dumpRawSkinSurface(const char* name=NULL);
        void dumpRawBorderSurface(const char* name=NULL);

 
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
        GSubstanceLib* getSubstanceLib();

    public:
        GMesh* getMesh(unsigned int index);  
        GMaterial* getMaterial(unsigned int index);  
        GSkinSurface* getSkinSurface(unsigned int index);  
        GBorderSurface* getBorderSurface(unsigned int index);  

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
        GMergedMesh* getMergedMesh(unsigned int index=0);

#if 0
    public:
        void materialConsistencyCheck();
        unsigned int materialConsistencyCheck(GSolid* solid);
#endif

    public:
        void Summary(const char* msg="GGeo::Summary");
        void Details(const char* msg="GGeo::Details");

    private:
        std::vector<GMesh*>           m_meshes ; 
        std::vector<GSolid*>          m_solids ; 
        std::vector<GMaterial*>       m_materials ; 
        std::vector<GSkinSurface*>    m_skin_surfaces ; 
        std::vector<GBorderSurface*>  m_border_surfaces ; 

        // _raw mainly for debug
        std::vector<GMaterial*>       m_materials_raw ; 
        std::vector<GSkinSurface*>    m_skin_surfaces_raw ; 
        std::vector<GBorderSurface*>  m_border_surfaces_raw ; 

        GSubstanceLib*                m_substance_lib ; 
        gfloat3*                      m_low ; 
        gfloat3*                      m_high ; 
        GMergedMesh*                  m_merged_mesh ; 

        char*                         m_path ;
        char*                         m_query ;
        char*                         m_ctrl ;
        char*                         m_idpath ;

    private:
        std::map<unsigned int, GSolid*>    m_solidmap ; 

};


inline GGeo::GGeo() :
   m_substance_lib(NULL),
   m_low(NULL),
   m_high(NULL),
   m_merged_mesh(NULL),
   m_path(NULL),
   m_query(NULL),
   m_ctrl(NULL),
   m_idpath(NULL)
{
   init(); 
}


inline void GGeo::add(GMaterial* material)
{
    m_materials.push_back(material);
}
inline void GGeo::add(GBorderSurface* surface)
{
    m_border_surfaces.push_back(surface);
}
inline void GGeo::add(GSkinSurface* surface)
{
    m_skin_surfaces.push_back(surface);
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


inline GSubstanceLib* GGeo::getSubstanceLib()
{
    return m_substance_lib ; 
}
inline gfloat3* GGeo::getLow()
{
   return m_low ; 
}
inline gfloat3* GGeo::getHigh()
{
   return m_high ; 
}




