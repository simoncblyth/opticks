#pragma once

#include "glm/fwd.hpp"

void print(const glm::mat4& m, const char* msg);
void print(const glm::vec3& v, const char* msg);
void print(const glm::vec4& v, const char* msg);
void print(const glm::vec4& a, const glm::vec4& b, const glm::vec4& c, const char* msg);
void print(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, const glm::vec3& d, const char* msg);
void print(const glm::quat& q, const char* msg);
void print(float* f, const char* msg, unsigned int n=16);

float absmax(glm::mat4& m);
void minmax(glm::mat4& m, float& mn, float& mx);


