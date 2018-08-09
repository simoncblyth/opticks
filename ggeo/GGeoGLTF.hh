#pragma once

#include <vector>
#include "GGEO_API_EXPORT.hh"

#include "GSolidRec.hh"

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

Writes the glTF 2.0 representation of a GGeo geometry.
glTF is a json based 3D file format that refers to other 
binary files for vertex and triangle data. 

Issues
-------

1. Suspect geocache duplication between the glTF extras and the GMeshLib persisted GMesh,
   TODO: consolidate to avoid this : not need to use an "extras" dir a "GMeshLib" 
   dir would work just fine  
   
2. Have not tried using this postcache OR with test geometry 
   
**/

class GGEO_API GGeoGLTF
{
    public:
        static void Save( const GGeo* ggeo, const char* path, int root ) ; 
    public:
        GGeoGLTF( const GGeo* ggeo ); 
        void save(const char* path, int root );
    private:
        void init();
        void addMaterials();
        void addMeshes();
        void addNodes();
    private:
        void addNodes_r(const GVolume* volume, YOG::Nd* parent_nd, int depth);
    public:
        void dumpSolidRec(const char* msg="GGeoGLTF::dumpSolidRec") const ;
        void writeSolidRec(const char* dir) const ;
    private:
        void solidRecTable( std::ostream& out ) const ; 
    private:
        const GGeo*            m_ggeo ;
        const GMaterialLib*    m_mlib ; 
        const GBndLib*         m_blib ; 
        YOG::Sc*               m_sc ;
        YOG::Maker*            m_maker ;
        std::vector<GSolidRec> m_solidrec ;

};


