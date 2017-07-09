#pragma once
#include <vector>

#include "NPY_API_EXPORT.hh"
#include "NGLM.hpp"
#include "Nuv.hpp"

struct NSceneConfig ; 
struct nnode ; 

class NPY_API NNodePoints
{
    public:
        NNodePoints(nnode* root, const NSceneConfig* config, float epsilon=1e-5);
        std::string desc() const ;
    public:
        glm::uvec4 collect_surface_points() ;
    public:
        const std::vector<glm::vec3>& getCompositePoints() const ;
        unsigned                      getNumCompositePoints() const ;
        float                         getEpsilon() const ;
        void dump(const char* msg="NNodePoints::dump", unsigned dmax=20) const  ;
    private:
        void init();
        glm::uvec4 collectCompositePoints( unsigned level, int margin , unsigned pointmask ) ;
        glm::uvec4 selectBySDF(const nnode* prim, unsigned prim_idx, unsigned pointmask ) ;



    private:
         nnode*                   m_root ; 
         const NSceneConfig*      m_config  ; 
         unsigned                 m_verbosity ; 
         float                    m_epsilon ;
         unsigned                 m_level ; 
         unsigned                 m_margin ; 
         unsigned                 m_target ; 

         std::vector<nnode*>       m_primitives ; 
         std::vector<glm::vec3>    m_composite_points ; 
         std::vector<nuv>          m_composite_coords ;  

};


