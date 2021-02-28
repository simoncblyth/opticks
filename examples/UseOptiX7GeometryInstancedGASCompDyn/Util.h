#pragma once
#include <vector>
#include <array>
#include <glm/glm.hpp>

struct Util
{

    static const char* PTXPath( const char* install_prefix, const char* cmake_target, const char* cu_stem, const char* cu_ext=".cu" );
    static void GetEyeUVW(const glm::vec3& eye_model, const glm::vec4& ce, const unsigned width, const unsigned height, glm::vec3& eye, glm::vec3& U, glm::vec3& V, glm::vec3& W );

    static void ParseGridSpec(       std::array<int,9>& grid, const char* spec ) ;
    static void GridMinMax(    const std::array<int,9>& grid, int& mn, int& mx ) ;

    template <typename T>
    static T ato_( const char* a );

    template <typename T>
    static T GetEValue(const char* key, T fallback);  

    template <typename T>
    static void GetEVector(std::vector<T>& vec, const char* key, const char* fallback );

    static void GetEVec(glm::vec3& v, const char* key, const char* fallback );


    template <typename T>
    static std::string Present(std::vector<T>& vec);


};
