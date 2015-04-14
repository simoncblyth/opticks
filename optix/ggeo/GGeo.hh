#ifndef GGEO_H
#define GGEO_H

#include <map>
#include <vector>
#include "GVector.hh"
#include "GDomain.hh"

class GMesh ; 
class GSolid ; 
class GNode ; 
class GMaterial ; 
class GSkinSurface ; 
class GBorderSurface ; 
class GSubstance ;
class GSubstanceLib ;
class GMergedMesh ;


class GGeo {
    public:
        GGeo();
        virtual ~GGeo();

    public:
        void setPath(const char* path);
        void setQuery(const char* query);
        void setCtrl(const char* ctrl);
        void setIdentityPath(const char* idpath);
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
        unsigned int getNumMeshes();
        unsigned int getNumSolids();
        unsigned int getNumMaterials();
        unsigned int getNumSkinSurfaces();
        unsigned int getNumBorderSurfaces();

    public:
        GSubstanceLib* getSubstanceLib();

    public:
        GMesh* getMesh(unsigned int index);  
        GMaterial* getMaterial(unsigned int index);  
        GSkinSurface* getSkinSurface(unsigned int index);  
        GBorderSurface* getBorderSurface(unsigned int index);  

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

    private:
        GSubstanceLib* m_substance_lib ; 
        std::vector<GMesh*>    m_meshes ; 
        std::vector<GSolid*>    m_solids ; 
        std::vector<GMaterial*> m_materials ; 
        std::vector<GSkinSurface*>  m_skin_surfaces ; 
        std::vector<GBorderSurface*>  m_border_surfaces ; 
        gfloat3* m_low ; 
        gfloat3* m_high ; 
        GMergedMesh* m_merged_mesh ; 

        char* m_path ;
        char* m_query ;
        char* m_ctrl ;
        char* m_idpath ;

    private:
        std::map<unsigned int, GSolid*>    m_solidmap ; 

};

#endif


