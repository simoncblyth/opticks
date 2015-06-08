#pragma once

#include "stdlib.h"
#include <vector>

class Renderer ; 
class Rdr ;
class Composition ; 
class NumpyEvt ; 
class GLoader ; 
class GDrawable ;
class GMergedMesh ;
class GGeo ;
class GSubstanceLibMetadata ;
class Photons ; 

#include "Configurable.hh"


class Scene : public Configurable {

   public:
       static const char* PHOTON ;
       static const char* GENSTEP ;
       static const char* GEOMETRY ;
       static const char* RECORD ;
   public:
       static const char* TARGET ;
   public:
       Scene();
       void gui();
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
        void setPhotons(Photons* photons);

   public:
        // target cannot live in Composition, as needs geometry 
        // to convert solid index into CenterExtent to give to Composition
        //
        void setTarget(unsigned int index=0); 
        unsigned int touch(int ix, int iy, float depth);
        void setTouch(unsigned int index); 
        unsigned int getTouch();
        void jump(); 

   public:
        const char* loadGeometry(const char* prefix, bool nogeocache=false);
        void uploadEvt();
        void render();

   public:
        unsigned int  getTarget(); 

   public:
        // pass throughs to the loader
        GMergedMesh*            getMergedMesh();
        GSubstanceLibMetadata*  getMetadata();
        GGeo*                   getGGeo();

   public:
        GLoader*      getGeometryLoader();
        Renderer*     getGeometryRenderer();
        Rdr*          getGenstepRenderer();
        Rdr*          getPhotonRenderer();
        Rdr*          getRecordRenderer();
        GDrawable*    getGeometry();
        Composition*  getComposition();
        NumpyEvt*     getNumpyEvt();
        Photons*      getPhotons();
        bool*         getModeAddress(const char* name);

   private:
        void init();

   private:
        GLoader*     m_geometry_loader ; 
        Renderer*    m_geometry_renderer ; 
        Rdr*         m_genstep_renderer ; 
        Rdr*         m_photon_renderer ; 
        Rdr*         m_record_renderer ; 
        NumpyEvt*    m_evt ;
        Photons*     m_photons ; 
        GDrawable*   m_geometry ;
        Composition* m_composition ;
        unsigned int m_target ;
        unsigned int m_touch ;
   private:
        bool         m_geometry_mode ; 
        bool         m_genstep_mode ; 
        bool         m_photon_mode ; 
        bool         m_record_mode ; 

};


inline Scene::Scene() :
            m_geometry_loader(NULL),
            m_geometry_renderer(NULL),
            m_genstep_renderer(NULL),
            m_photon_renderer(NULL),
            m_record_renderer(NULL),
            m_evt(NULL),
            m_photons(NULL),
            m_geometry(NULL),
            m_composition(NULL),
            m_target(0),
            m_touch(0),
            m_geometry_mode(true),
            m_genstep_mode(true),
            m_photon_mode(true),
            m_record_mode(true)
{
    init();
}



inline unsigned int Scene::getTarget()
{
    return m_target ;
}
inline unsigned int Scene::getTouch()
{
    return m_touch ;
}
inline void Scene::setTouch(unsigned int touch)
{
    m_touch = touch ; 
}


inline GLoader* Scene::getGeometryLoader()
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
inline Rdr* Scene::getRecordRenderer()
{
    return m_record_renderer ; 
}





inline NumpyEvt* Scene::getNumpyEvt()
{
    return m_evt ; 
}
inline Photons* Scene::getPhotons()
{
    return m_photons ; 
}




inline GDrawable* Scene::getGeometry()
{
    return m_geometry ; 
}
inline void Scene::setNumpyEvt(NumpyEvt* evt)
{
    m_evt = evt ; 
}
inline void Scene::setPhotons(Photons* photons)
{
    m_photons = photons ; 
}




