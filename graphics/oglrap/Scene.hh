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
class GSubstanceLibMetadata ;

#include "Configurable.hh"

class Scene : public Configurable {
   public:
        static const char* TARGET ;
        Scene();
   public:
        // Configurable
        std::vector<std::string> getTags();
        void set(const char* name, std::string& xyz);
        std::string get(const char* name);

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
        unsigned int touch(int ix, int iy, float depth);

   public:
        const char* loadGeometry(const char* prefix, bool nogeocache=false);
        void loadEvt();
        void render();

   public:
        unsigned int  getTarget(); 

   public:
        // pass throughs to the loader
        GMergedMesh*            getMergedMesh();
        GSubstanceLibMetadata*  getMetadata();
        GGeo*                   getGGeo();

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


inline Scene::Scene() :
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



