#pragma once

#include "glm/fwd.hpp"
#include <string>

#include "NPY_API_EXPORT.hh"

NPY_API void fdump(float* f, unsigned int n=16, const char* msg="fdump");

NPY_API std::string format(int i);
NPY_API std::string format(float f);
NPY_API std::string format(const glm::vec3& v );

NPY_API void print(const glm::mat4& m, const char* msg);
NPY_API void print(const glm::vec2& v, const char* msg);
NPY_API void print(const glm::vec3& v, const char* msg);
NPY_API void print(const glm::vec4& v, const char* msg);
NPY_API void print(const glm::vec4& v0, const char* msg0, const glm::vec4& v1, const char* msg1);
NPY_API void print(const glm::vec4& v, const char* tmpl, unsigned int incl);
//NPY_API void print_i(const glm::ivec4& v, const char* msg);
//NPY_API void print_u(const glm::uvec4& v, const char* msg);

NPY_API void print(const glm::ivec4& v, const char* msg);
NPY_API void print(const glm::uvec4& v, const char* msg);



NPY_API void print(const glm::vec4& a, const glm::vec4& b, const glm::vec4& c, const char* msg);
NPY_API void print(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, const glm::vec3& d, const char* msg);
NPY_API void print(const glm::quat& q, const char* msg);
NPY_API void print(float* f, const char* msg, unsigned int n=16);

NPY_API float absmax(glm::mat4& m);
NPY_API void minmax(glm::mat4& m, float& mn, float& mx);

NPY_API void assert_same(const char* msg, const glm::vec4& a, const glm::vec4& b);

