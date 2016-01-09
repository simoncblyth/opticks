#include <iostream>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

void dirlist(std::vector<std::string>& basenames,  const char* path, const char* ext)
{
    fs::path dir(path);
    if(!( fs::exists(dir) && fs::is_directory(dir))) return ; 

    fs::directory_iterator it(dir) ;
    fs::directory_iterator end ;
   
    for(; it != end ; ++it)
    {
        std::string fname = it->path().filename().string() ;
        const char* fnam = fname.c_str();

        if(strlen(fnam) > strlen(ext) && strcmp(fnam + strlen(fnam) - strlen(ext), ext)==0)
        {
            std::string basename(fnam, fnam+strlen(fnam)-strlen(ext));
            //std::cout << basename << std::endl ;
            basenames.push_back(basename);
        }
    }
}


void dirlist(std::vector<std::string>& names,  const char* path)
{
    fs::path dir(path);
    if(!( fs::exists(dir) && fs::is_directory(dir))) return ; 

    fs::directory_iterator it(dir) ;
    fs::directory_iterator end ;
   
    for(; it != end ; ++it)
    {
        std::string name = it->path().filename().string() ;
        //std::cout << name << std::endl ;
        names.push_back(name);
    }
}

