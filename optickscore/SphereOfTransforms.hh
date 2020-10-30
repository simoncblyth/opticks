#pragma once
#include <string>
#include "plog/Severity.h"
#include "NGLM.hpp"

template <typename> class NPY ;

#include "OKCORE_API_EXPORT.hh"

class OKCORE_API SphereOfTransforms {
    private:
        static const plog::Severity LEVEL ; 
    public:
        static NPY<float>* Make(float radius, unsigned num_theta, unsigned num_phi);
    public:
        SphereOfTransforms(float radius, unsigned num_theta, unsigned num_phi); 
        std::string desc() const;
        NPY<float>* getTransforms() const ; 
    private:
        void init(); 
        void get_pos_nrm( glm::vec3& pos, glm::vec3& nrm, float fphi, float ftheta ) const ;
    private:
        float       m_radius ; 
        unsigned    m_num_theta ; 
        unsigned    m_num_phi ; 
        unsigned    m_num_transforms ;
        NPY<float>* m_transforms ;  
};



