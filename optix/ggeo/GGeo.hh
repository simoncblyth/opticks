#ifndef GGEO_H
#define GGEO_H

#include <vector>

class GMesh ; 
class GSolid ; 
class GMaterial ; 
class GSkinSurface ; 
class GBorderSurface ; 

class GGeo {
    public:
        GGeo();
        virtual ~GGeo();

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
        GMesh* getMesh(unsigned int index);  
        GMaterial* getMaterial(unsigned int index);  
        GSolid* getSolid(unsigned int index);  
        GSkinSurface* getSkinSurface(unsigned int index);  
        GBorderSurface* getBorderSurface(unsigned int index);  

    public:
        GSkinSurface* findSkinSurface(const char* lv);  
        GBorderSurface* findBorderSurface(const char* pv1, const char* pv2);  

    public:
        void materialConsistencyCheck();
        unsigned int materialConsistencyCheck(GSolid* solid);

    public:
        void Summary(const char* msg="GGeo::Summary");

    private:
        std::vector<GMesh*>    m_meshes ; 
        std::vector<GSolid*>    m_solids ; 
        std::vector<GMaterial*> m_materials ; 
        std::vector<GSkinSurface*>  m_skin_surfaces ; 
        std::vector<GBorderSurface*>  m_border_surfaces ; 

        unsigned int m_check ;

};

#endif


