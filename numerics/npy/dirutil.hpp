#pragma once

#include <vector>
#include <string>

void dirlist(std::vector<std::string>& names,  const char* path);
void dirlist(std::vector<std::string>& basenames,  const char* path, const char* ext);

// basenames of directories within the path directory
void dirdirlist(std::vector<std::string>& names,  const char* path);


