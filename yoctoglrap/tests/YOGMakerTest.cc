
#include <string>
#include <iostream>
#include <fstream>

#include "YOGMaker.hh"
#include "YOGGeometry.hh"

int main(int argc, char** argv)
{
    YOGGeometry geom ; 
    geom.make_triangle();

    auto gltf = YOGMaker::demo_make_gltf( geom ) ; 

    std::string path = "/tmp/YOGMakerTest/YOGMakerTest.gltf" ; 
    bool save_bin = true ; 
    bool save_shaders = false ; 
    bool save_images = false ; 

    save_gltf(path, gltf.get(), save_bin, save_shaders, save_images);

    std::cout << "writing " << path << std::endl ; 

    std::ifstream fp(path);
    std::string line;
    while(std::getline(fp, line)) std::cout << line << std::endl ; 
    return 0 ; 
}


