#pragma once

#include "stdlib.h"
#include <vector>

class Renderer ; 
class Rdr ;
class Composition ; 
class NumpyEvt ; 
class Geometry ; 
class GDrawable ;
class GMergedMesh ;
class GGeo ;

class Scene {
   public:
        static const char* TARGET ;

        Scene() :
            m_geometry_loader(NULL),
            m_geometry_renderer(NULL),
            m_genstep_renderer(NULL),
            m_photon_renderer(NULL),
            m_evt(NULL),
            m_geometry(NULL),
            m_composition(NULL),
            m_target(0)
        {
            init();
        }

   public:
        static bool accepts(const char* name);
        void configure(const char* name, const char* value_);
        void configure(const char* name, int value);

   public:
        void configureI(const char* name, std::vector<int> values);
        void setComposition(Composition* composition);
        void setNumpyEvt(NumpyEvt* evt);

   public:
        // target cannot live in Composition, as needs geometry 
        // to convert solid index into CenterExtent to give to Composition
        //
        void setTarget(unsigned int index=0); 
        void touch(unsigned char key, int ix, int iy, float depth);

   public:
        const char* loadGeometry(const char* prefix, bool nogeocache=false);
        void loadEvt();
        void render();

   public:
        GMergedMesh*  getMergedMesh();
        GGeo*         getGGeo();
        unsigned int  getTarget(); 

   public:
        Geometry*     getGeometryLoader();
        Renderer*     getGeometryRenderer();
        Rdr*          getGenstepRenderer();
        Rdr*          getPhotonRenderer();
        GDrawable*    getGeometry();
        Composition*  getComposition();
        NumpyEvt*     getNumpyEvt();

   private:
        void init();

   private:
        Geometry*    m_geometry_loader ; 
        Renderer*    m_geometry_renderer ; 
        Rdr*         m_genstep_renderer ; 
        Rdr*         m_photon_renderer ; 
        NumpyEvt*    m_evt ;
        GDrawable*   m_geometry ;
        Composition* m_composition ;
        unsigned int m_target ;

};



inline unsigned int Scene::getTarget()
{
    return m_target ;
}
inline Geometry* Scene::getGeometryLoader()
{
    return m_geometry_loader ; 
}
inline Renderer* Scene::getGeometryRenderer()
{
    return m_geometry_renderer ; 
}
inline Rdr* Scene::getGenstepRenderer()
{
    return m_genstep_renderer ; 
}
inline Rdr* Scene::getPhotonRenderer()
{
    return m_photon_renderer ; 
}
inline NumpyEvt* Scene::getNumpyEvt()
{
    return m_evt ; 
}
inline GDrawable* Scene::getGeometry()
{
    return m_geometry ; 
}
inline void Scene::setNumpyEvt(NumpyEvt* evt)
{
    m_evt = evt ; 
}





