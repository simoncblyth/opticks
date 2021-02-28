#pragma once

#include <array>
#include <vector>
#include <glm/glm.hpp>

struct Grid
{
    unsigned               num_shape ; 
    std::array<int,9>      grid ; 
    std::vector<unsigned>  shape_modulo ;  
    std::vector<unsigned>  shape_single ;  
    std::vector<glm::mat4> trs ;  

    Grid(unsigned num_shape);
    int extent() const ;
    std::string desc() const ;

    void init(); 
    void write(const char* base, const char* rel, unsigned idx ) const ;
};


