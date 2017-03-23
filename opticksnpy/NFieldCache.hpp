#pragma once

#include <string>
#include <functional>
#include  <boost/unordered_map.hpp>
#include "NBBox.hpp"

#include "NPY_API_EXPORT.hh"

class NPY_API NFieldCache {
    public:
         typedef boost::unordered_map<unsigned int, float> UMAP ; 
         NFieldCache(std::function<float(float,float,float)> field, const nbbox& bb);
         float operator()(float x, float y, float z);
         unsigned getMortonCode(float x, float y, float z);
         std::string desc();
         void reset();
    private:
         std::function<float(float,float,float)>   m_field ;
         nbbox m_bbox ; 
         nvec3 m_side ; 
         UMAP m_cache;
         unsigned m_calc ; 
         unsigned m_lookup ; 
  
};
