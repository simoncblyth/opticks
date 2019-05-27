//  https://www.boost.org/doc/libs/1_49_0/libs/filesystem/v3/doc/tutorial.html

#include <iostream>
#include "UseBoost.hh"




int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: UseUseBoost path elememts to be joined into a path to file.txt\n";
        UseBoost::dump_file_size(argv[0]);
        return 0 ;
    }

    const char* path = UseBoost::concat_path( argc, argv ); 
    std::cout << " path " << path << std::endl ; 


    UseBoost::dump_file_size(path);

    return 0;
}


