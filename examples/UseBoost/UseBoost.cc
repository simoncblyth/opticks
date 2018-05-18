//  https://www.boost.org/doc/libs/1_49_0/libs/filesystem/v3/doc/tutorial.html

#include <iostream>
#include <boost/filesystem.hpp>
using namespace boost::filesystem;


void dump_file_size(const char* path)
{
    std::cout << "dump_file_size: \"" << path  << "\" " << file_size(path) << '\n';
}


