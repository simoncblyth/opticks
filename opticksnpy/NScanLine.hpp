#pragma once

struct nnode ; 

#include <vector>
#include <string>
#include "NGLM.hpp"

#include "NPY_API_EXPORT.hh"

class NPY_API NScanLine {
    public:

        float path_length() const ; 
        float step_length() const ; 
        unsigned num_step() const ; 
        unsigned get_num_step() const ; 
        const std::string& get_message() const ; 

        glm::vec3 make_step(const glm::vec3& step) const ;

        NScanLine( const glm::vec3& begin,  const glm::vec3& end, const glm::vec3& step, unsigned verbosity);
        void setNode(const nnode* node);
        void setNode_(const nnode* node);
        void setNodes(const std::vector<const nnode*>& nodes);

        void dump( const char* msg, int t0=0, int t1=-1) ;


        std::string desc() const  ; 
        std::string desc_zeros() const ;

        void add_zero(const glm::uvec4& tbrak ) ;
        bool has_zero(const glm::uvec4& brak) const ;
        bool has_zero( unsigned t0 ) const ;


        glm::vec3 position(int step) const ;
        float sdf(int step) const  ;
        float sdf0() const  ;
        float sdf1() const  ;

        void find_zeros();
        void find_zeros_one();
        unsigned count_zeros(unsigned node_idx_) const;

        void sample(std::vector<float>& sd) const ;



    private:
        const glm::vec3& m_begin ;
        const glm::vec3& m_end  ;
        const glm::vec3  m_path ;
        const glm::vec3  m_step ;
        const unsigned   m_verbosity ; 
        const unsigned   m_num_step ; 

        const nnode* m_node ; 
        std::vector<const nnode*> m_nodes ; 
        unsigned m_num_nodes ; 

        std::vector<glm::uvec4> m_zeros ; 

        std::string m_message ; 

};



