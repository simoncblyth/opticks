#pragma once

#include <vector>
#include <functional>
#include <boost/unordered_map.hpp>

#include "NGLM.hpp"
#include "NQuad.hpp"
#include "NBBox.hpp"
#include "NGrid3.hpp"

struct NGrid3 ; 
struct NField3 ; 
struct NFieldGrid3 ; 

class Timer ; 
class NTrianglesNPY ; 

#include "NPY_API_EXPORT.hh"


struct NPY_API NMeshVertex
{
	NMeshVertex(const glm::vec3& _xyz, const glm::vec3& _normal)
		: xyz(_xyz)
		, normal(_normal)
	{
	}

	glm::vec3		xyz, normal;
};



typedef std::vector<NMeshVertex> NVertexBuffer;
typedef std::vector<int> NIndexBuffer;


enum { 
               BUILD_BOTTOM_UP = 0x1 << 0, 
               BUILD_TOP_DOWN  = 0x1 << 1,
               USE_BOTTOM_UP   = 0x1 << 2, 
               USE_TOP_DOWN    = 0x1 << 3, 
               BUILD_BOTH      = BUILD_BOTTOM_UP | BUILD_TOP_DOWN
      };
 

template<typename T>
class NPY_API NConstructor 
{
    static const int maxlevel = 10 ; 
    static const nivec3 _CHILD_MIN_OFFSETS[8] ;

    typedef std::function<float(float,float,float)> FN ; 
    typedef boost::unordered_map<unsigned, T*> UMAP ;
    UMAP cache[maxlevel] ; 

    public:
        NConstructor(NFieldGrid3* fieldgrid, const nvec4& ce, const nbbox& bb, int nominal, int coarse, int verbosity );
        T* create();
        void dump();
        void report(const char* msg="NConstructor::report");
    public:
         // grid debugging 
        void scan(const char* msg="scan", int depth=2, int limit=30 ) const ; 
        void corner_scan(const char* msg="corner_scan", int depth=2, int limit=30) const ; 

        void dump_domain(const char* msg="dump_domain") const ;

        nvec3 position_ce(const nivec3& offset_ijk, int depth) const ;
        float density_ce(const nivec3& offset_ijk, int depth) const ;

        nvec3 position_bb(const nivec3& natural_ijk, int depth) const ;
        float density_bb(const nivec3& natural_ijk, int depth) const ;
    private:
        T* make_leaf(const nivec3& min, int leaf_size, int corners );
        T* create_coarse_nominal();
        T* create_nominal();
        void buildBottomUpFromLeaf(int leaf_loc, T* leaf );
    private:
        NMultiGrid3 m_mgrid ; 

        NFieldGrid3* m_fieldgrid ; 
        NField3*    m_field ; 
        FN*         m_func ; 
        nvec4       m_ce ;  
        nbbox       m_bb ; 

        NGrid3*     m_nominal ; 
        NGrid3*     m_coarse ; 
        int         m_verbosity ; 
        NGrid3*     m_subtile ; 
        NGrid3*     m_dgrid ; 

        nivec3      m_nominal_min ; 
        int         m_upscale_factor ; 

        T* m_root ; 

        unsigned m_num_leaf ; 
        unsigned m_num_from_cache ; 
        unsigned m_num_into_cache ; 
        unsigned m_coarse_corners ; 
        unsigned m_nominal_corners ; 
     
};




template<typename T>
class NPY_API NManager 
{
    public:
   public:
        NManager(const unsigned ctrl, const int nominal, const int coarse, const int verbosity, const float threshold, NFieldGrid3* fieldgrid, const nbbox& bb, Timer* timer);

        void buildOctree();
        void generateMeshFromOctree();
        NTrianglesNPY* collectTriangles();
        void meshReport(const char* msg="NManager::meshReport");

    private:
        unsigned m_ctrl ; 
        int      m_nominal_size ; 
        int      m_verbosity ; 
        float    m_threshold ;
        NFieldGrid3* m_fieldgrid ; 
        nbbox    m_bb ; 
        Timer*   m_timer ;    

        nvec4        m_ce ; 
        NConstructor<T>* m_ctor ; 

        T*  m_bottom_up ;         
        T*  m_top_down ;         
        T*  m_raw ;         
        T*  m_simplified ;         


        std::vector<glm::vec3> m_vertices;
        std::vector<glm::vec3> m_normals;
        std::vector<int>       m_indices;


};


