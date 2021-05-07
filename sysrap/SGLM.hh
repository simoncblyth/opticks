#pragma once

#include <string>
#include <vector>
#include <glm/fwd.hpp>

#include "SYSRAP_API_EXPORT.hh"

struct SYSRAP_API SGLM
{
    template <typename T>
    static T ato_( const char* a );

    template <typename T>
    static void GetEVector(std::vector<T>& vec, const char* key, const char* fallback );

    template <typename T>
    static std::string Present(std::vector<T>& vec);

    static std::string Present(glm::vec3& v, int wid=10, int prec=3);
    static std::string Present(glm::vec4& v, int wid=10, int prec=3);

    static void GetEVec(glm::vec3& v, const char* key, const char* fallback );
    static void GetEVec(glm::vec4& v, const char* key, const char* fallback );

    static glm::vec3 EVec3(const char* key, const char* fallback); 
    static glm::vec4 EVec4(const char* key, const char* fallback); 

};





