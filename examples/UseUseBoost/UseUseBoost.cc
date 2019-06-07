//  https://www.boost.org/doc/libs/1_49_0/libs/filesystem/v3/doc/tutorial.html

#include <iostream>
#include "UseBoost.hh"

#include <boost/version.hpp>


int main(int argc, char** argv)
{
    const char* path = argc > 1 ? argv[1] : argv[0] ; 

    //UseBoost::dump_file_size(argv[0]);

    //const char* path = UseBoost::concat_path( argc, argv ); 
    //std::cout << " path " << path << std::endl ; 

    //UseBoost::dump_file_size(path);


    std::cout 
          << "Using Boost "     
          << BOOST_VERSION / 100000     << "."  // major version
          << BOOST_VERSION / 100 % 1000 << "."  // minor version
          << BOOST_VERSION % 100                // patch level
          << std::endl;


    return 0;
}


