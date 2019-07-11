#pragma once

#include "OGLRAP_API_EXPORT.hh"
#include "plog/Severity.h"
#include <glm/glm.hpp>


struct InstLODCullContext ;

struct OGLRAP_API RContext
{
    static const plog::Severity LEVEL ; 
    static const char* uniformBlockName ;

    InstLODCullContext*   uniform ; 
    GLuint                uniformBO ; 
   
    RContext();

    void init();

    void initUniformBuffer();
    void bindUniformBlock(GLuint program);

    void update( const glm::mat4& world2clip, const glm::mat4& world2eye, const glm::vec4& lodcut );

};
