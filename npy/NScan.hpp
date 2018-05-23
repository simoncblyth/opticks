#pragma once

#include <string>
#include <vector>

#include "NGLM.hpp"

#include "NPY_API_EXPORT.hh"
#include "NBBox.hpp"

struct nnode ; 


/*

NScan
=======

See issues/NScanTest_csg_zero_crossings.rst 


Touching unions will just hit zero and then go negative again
so need to brute force scan to find these ... or maybe should nudge to encourage
a sign change, to look for candidate regions.

Along any scan segment finding roots is simpler and more reliable for primitives
as those will have clear sign transitions.
So scan the primitives collecting zero brackets to be
checked/bisected/brute-forced on the composite.

*/


class NPY_API  NScan 
{
    public:
        bool has_message() const ; 
        const std::string& get_message() const ; 
        unsigned     get_nzero() const ; 
        void set_nzero(unsigned nzero) ;
        static void init_cage(const nbbox& bb, glm::vec3& bmin, glm::vec3& bmax, glm::vec3& bcen, float sidescale, float minmargin, unsigned verbosity ) ;
        NScan( const nnode& node, unsigned verbosity );
        void scan(std::vector<float>& sd, const glm::vec3& origin, const glm::vec3& direction, const glm::vec3& tt );
        unsigned autoscan(float mmstep);
        std::string desc() const  ; 
    private:
        void init(); 
    private:
        const nnode& m_node ;
        const nbbox  m_bbox ; 
        unsigned     m_verbosity ; 
        unsigned     m_num_prim ; 
        unsigned     m_num_nodes ; 

        std::vector<const nnode*> m_prim ; 
        std::vector<const nnode*> m_nodes ; 
        std::vector<std::string>  m_tags ; 


        glm::vec3    m_bmin ; 
        glm::vec3    m_bmax ; 
        glm::vec3    m_bcen ; 

        std::string  m_message ; 
        unsigned     m_nzero ; 

};
 


