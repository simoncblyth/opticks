#include <boost/filesystem.hpp>
#include <iostream>

namespace fs = boost::filesystem;



int main(int argc, char** argv)
{
    typedef fs::directory_iterator Di ;
    typedef fs::directory_entry    De ;

    for(Di i=Di(argv[1]) ; i!=Di() ; i++)
    {
        De entry = *i ;

        fs::path p = entry.path();


        std::cout << entry << " : [" << p.string()  <<  "]" << std::endl; 
    } 

}
