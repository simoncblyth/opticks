#pragma once

#include <map>
#include <string>
#include <vector>
#include <ostream>

class Opticks ; 

class GItemIndex ; 
class GMesh ; 


#include "GGEO_API_EXPORT.hh"

/*

GMeshLib : provides load/save for GMesh instances with associated names
==========================================================================

* canonical m_meshlib instances are constituents of GGeo and GScene.
* manages a vector of GMesh* and a GItemIndex of names
* not on critical path (?) : used in checking feasibility of a polygonization implementation  



TODO: use GItemList rather than GItemIndex for the names
-------------------------------------------------------------

The local/source indices are 1-based and 0-based, thats not a good 
reason to use an Index.

::

    epsilon:MeshIndex blyth$ ll
    total 48
    -rw-r--r--   1 blyth  staff  9448 Jun 28 10:45 GItemIndexSource.json
    -rw-r--r--   1 blyth  staff  9450 Jun 28 10:45 GItemIndexLocal.json
    drwxr-xr-x  17 blyth  staff   544 Jun 28 10:45 ..

::

    epsilon:MeshIndex blyth$ diff -y  GItemIndexLocal.json GItemIndexSource.json | head -10
    {								{
        "AcrylicCylinder0xc3d3830": "137",			  |	    "AcrylicCylinder0xc3d3830": "136",
        "AdPmtCollar0xc2c5260": "49",			      |	    "AdPmtCollar0xc2c5260": "48",
        "AmCCo60AcrylicContainer0xc0b23b8": "132",	  |	    "AmCCo60AcrylicContainer0xc0b23b8": "131",
        "AmCCo60Cavity0xc0b3de0": "131",			  |	    "AmCCo60Cavity0xc0b3de0": "130",
        "AmCCo60SourceAcrylic0xc3ce678": "123",		  |	    "AmCCo60SourceAcrylic0xc3ce678": "122",
        "AmCSS0xc3d0040": "121",				      |	    "AmCSS0xc3d0040": "120",
        "AmCSSCap0xc3cfc58": "116",				      |	    "AmCSSCap0xc3cfc58": "115",
        "AmCSource0xc3d0708": "120",			      |	    "AmCSource0xc3d0708": "119",
        "AmCSourceAcrylicCup0xc3d1bc8": "119",		  |	    "AmCSourceAcrylicCup0xc3d1bc8": "118",




  
::

    op --dsst -G 

*/

class GGEO_API GMeshLib 
{
        friend class GGeo ; 
        friend class GScene ; 
    public:
        static const unsigned MAX_MESH  ; 

        static const char*    GITEMINDEX ; 
        static const char*    GMESHLIB_INDEX ; 
        static const char*    GMESHLIB_INDEX_ANALYTIC ; 
        static const char*    GetRelDirIndex(bool analytic);

        static const char*    GMESHLIB ; 
        static const char*    GMESHLIB_ANALYTIC ; 
        static const char*    GetRelDir(bool analytic);

        static GMeshLib* Load(Opticks* ok, bool analytic);
    public:
        GMeshLib(Opticks* opticks, bool analytic); 
        bool isAnalytic() const ; 
        void add(const GMesh* mesh);
        void dump(const char* msg="GMeshLib::dump") const;
    public:
        // methods working from the index, so work prior to loading meshes
        unsigned    getMeshIndex(const char* name, bool startswith) const ;
        const char* getMeshName(unsigned aindex) ; 
    public:
        //std::string desc() const ; 
        GItemIndex* getMeshIndex() ;
        unsigned    getNumMeshes() const ; 
        const GMesh*  getMesh(unsigned aindex) const ;
        const GMesh*  getMesh(const char* name, bool startswith) const ;
    private:
        void        loadFromCache();
        void        save() const ; 
    private:
        void saveMeshes(const char* idpath) const ;
        void loadMeshes(const char* idpath ) ;
        void removeMeshes(const char* idpath ) const ;

    public:
        std::map<unsigned,unsigned>& getMeshUsage();
        std::map<unsigned,std::vector<unsigned> >& getMeshNodes();
        void countMeshUsage(unsigned meshIndex, unsigned nodeIndex);
        void reportMeshUsage(const char* msg="GMeshLib::reportMeshUsage") const ;
        void writeMeshUsage(const char* path="/tmp/GMeshLib_MeshUsageReport.txt") const ;
        void reportMeshUsage_(std::ostream& out) const ;
        void saveMeshUsage(const char* idpath) const ;
    private:
        Opticks*                      m_ok ; 
        bool                          m_analytic ; 
        const char*                   m_reldir ; 
        GItemIndex*                   m_meshindex ; 
        unsigned                      m_missing ; 
        std::vector<const GMesh*>     m_meshes ; 


        std::map<unsigned, unsigned>                  m_mesh_usage ; 
        std::map<unsigned, std::vector<unsigned> >    m_mesh_nodes ; 


};
