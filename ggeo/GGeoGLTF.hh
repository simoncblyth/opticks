#pragma once

#include "GGEO_API_EXPORT.hh"

class GGeo ; 
class GVolume ; 
class GMaterialLib ; 
class GBndLib ; 

namespace YOG 
{
   struct Sc ; 
   struct Mh ; 
   struct Nd ; 
   struct Maker ; 
}

/**
GGeoGLTF
=========

TODO: const GGeo

**/

class GGEO_API GGeoGLTF
{
    public:
        static void Save( GGeo* ggeo, const char* path, int root ) ; 
    public:
        GGeoGLTF( GGeo* ggeo ); 
        void save(const char* path, int root );
    private:
        void init();
        void addMaterials();
        void addMeshes();
        void addNodes();
    private:
        void addNodes_r(const GVolume* volume, YOG::Nd* parent_nd, int depth);
    private:
        const GGeo*          m_ggeo ;
        const GMaterialLib*  m_mlib ; 
        const GBndLib*       m_blib ; 
        YOG::Sc*             m_sc ;
        YOG::Maker*          m_maker ;

};


