//  https://www.boost.org/doc/libs/1_49_0/libs/filesystem/v3/doc/tutorial.html

#include "UseBoost.hh"

#include <iostream>
#include <string>
#include <cstring>

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;


void UseBoost::dump_file_size(const char* path)
{
    std::cout << "dump_file_size: \"" << path  << "\" " << fs::file_size(path) << '\n';
}


const char* UseBoost::concat_path(int argc, char** argv)
{
    fs::path p ;
    for(int i=1 ; i < argc ; i++)
    {
         char* a = argv[i] ; 
         if(a) p /= a ; 
    }

    std::string x = p.string() ;
    return strdup(x.c_str());
}


/*
const char* UseBoost::program_location()
{

//#include <boost/dll/runtime_symbol_info.hpp>

   
https://www.boost.org/doc/libs/1_70_0/doc/html/boost_dll/reference.html#header.boost.dll.runtime_symbol_info_hpp    

https://stackoverflow.com/questions/1528298/get-path-of-executable



    std::string x = p.string() ;
    return strdup(x.c_str());
}

*/  


