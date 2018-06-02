
#include <string>
#include <iostream>
#include <fstream>

#include "YOGMaker.hh"

int main(int argc, char** argv)
{
    //auto gltf = YOGMaker::make_gltf() ; 
    auto gltf = YOGMaker::make_gltf_example() ; 

    std::string path = "/tmp/UseYoctoGL_Write.gltf" ; 
    bool save_bin = false ; 
    bool save_shaders = false ; 
    bool save_images = false ; 

    save_gltf(path, gltf.get(), save_bin, save_shaders, save_images);

    std::cout << "writing " << path << std::endl ; 

    std::ifstream fp(path);
    std::string line;
    while(std::getline(fp, line)) std::cout << line << std::endl ; 
    return 0 ; 
}


