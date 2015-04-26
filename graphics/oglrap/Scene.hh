#pragma once

#include "stdlib.h"

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
        Scene() :
            m_geometry_loader(NULL),
            m_geometry_renderer(NULL),
            m_genstep_renderer(NULL),
            m_photon_renderer(NULL),
            m_evt(NULL),
            m_geometry(NULL),
            m_composition(NULL),
            m_target(NULL)
        {
            init();
        }

   public:
        void setComposition(Composition* composition);
        void setNumpyEvt(NumpyEvt* evt);

   public:
        void loadGeometry(const char* prefix);
        void loadEvt();
        void render();
 

   public:
        GMergedMesh*  getMergedMesh();
        GGeo*         getGGeo();

   public:
        Geometry*     getGeometryLoader();
        Renderer*     getGeometryRenderer();
        Rdr*          getGenstepRenderer();
        Rdr*          getPhotonRenderer();
        GDrawable*    getGeometry();
        Composition*  getComposition();
        NumpyEvt*     getNumpyEvt();

   private:
        void setTarget(unsigned int index=0); // better in Composition
        float*        getTarget(); // better in Composition
        void init();

   private:
        Geometry*    m_geometry_loader ; 
        Renderer*    m_geometry_renderer ; 
        Rdr*         m_genstep_renderer ; 
        Rdr*         m_photon_renderer ; 
        NumpyEvt*    m_evt ;
        GDrawable*   m_geometry ;
        Composition* m_composition ;
        float*       m_target ;

};




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







