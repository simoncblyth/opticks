#pragma once
#include <glm/glm.hpp>

struct Util
{
    static const char* PTXPath( const char* install_prefix, const char* cmake_target, const char* cu_stem, const char* cu_ext=".cu" );
    static const char* PPMPath( const char* install_prefix, const char* stem, const char* ext=".ppm" ); 
    static void GetEyeUVW(const glm::vec4& ce, const unsigned width, const unsigned height, glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W );

    template <typename T>
    static T ato_( const char* a );

    template <typename T>
    static T GetEValue(const char* key, T fallback);  

};
