#pragma once
#include <vector>

#include "NPY_API_EXPORT.hh"
#include "NGLM.hpp"
#include "Nuv.hpp"

struct NSceneConfig ; 
struct nnode ; 
struct nbbox ; 

class NPY_API NNodePoints
{
    public:
        NNodePoints(nnode* root, const NSceneConfig* config=NULL );
        std::string desc() const ;
        void setEpsilon(float epsilon); 
    public:
        glm::uvec4 collect_surface_points() ;
        nbbox bbox_surface_points() const  ;
    public:
        nbbox selectPointsBBox( unsigned prim, unsigned sheet ) const ;
        void selectPoints(std::vector<glm::vec3>& points, std::vector<nuv>& coords, unsigned prim, unsigned sheet) const ;

        const std::vector<glm::vec3>& getCompositePoints() const ;
        unsigned                      getNumCompositePoints() const ;
        float                         getEpsilon() const ;
        void dump(const char* msg="NNodePoints::dump", unsigned dmax=20) const  ;
    private:
        void init();
        void clear();

        glm::uvec4 collectCompositePoints( unsigned level, int margin , unsigned pointmask ) ;
        glm::uvec4 selectBySDF(const nnode* prim, unsigned prim_idx, unsigned pointmask ) ;
        void dump_sheets() const ;
        void dump_bb() const ;

    private:
        nnode*                   m_root ; 
        const NSceneConfig*      m_config  ; 
        unsigned                 m_verbosity ; 
        float                    m_epsilon ;
        unsigned                 m_level ; 
        unsigned                 m_margin ; 
        unsigned                 m_target ; 

        std::vector<nnode*>       m_primitives ; 
        std::vector<nbbox>        m_prim_bb ; 
        std::vector<nbbox>        m_prim_bb_selected ; 
        std::vector<glm::vec3>    m_composite_points ; 
        std::vector<nuv>          m_composite_coords ;  

};


