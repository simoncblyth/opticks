#pragma once

#include <string>
#include <set>
#include <vector>
#include "plog/Severity.h"

#include "NGLM.hpp"
template<typename T> class NPY ; 
#include "OKCORE_API_EXPORT.hh"

class OKCORE_API IntersectSDF {
    public:
        IntersectSDF(const char* dir, float epsilon=4e-4); 
        unsigned getRC() const ; 
        std::string desc() const ;
    private:
        friend struct IntersectSDFTest ;   
        static const plog::Severity LEVEL ;  
        float sdf(unsigned geocode, const glm::vec3& lpos );
        static NPY<unsigned>* ExtractTransformIdentity( const NPY<float>* transforms);  
        static NPY<unsigned>* ExtractUInt( const NPY<float>* src, unsigned j, unsigned k ); 
        static unsigned FixColumnFour( NPY<float>* a );
        void check_lpos_sdf();
        void select_intersect_tranforms(std::set<unsigned>& tpx, unsigned geocode);
        void get_local_intersects(std::vector<glm::vec4>& lpos, unsigned transform_index);
    private:
        const char*         m_dir ; 
        float               m_epsilon ; 
        NPY<unsigned char>* m_pixels ; 
        NPY<float>*         m_posi ; 
        NPY<float>*         m_transforms ; 
        NPY<unsigned>*      m_identity ; 
        unsigned            m_fixcount ; 
        NPY<float>*         m_itransforms ; 
        unsigned            m_rc ; 

}; 
