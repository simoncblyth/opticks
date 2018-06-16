//  https://www.boost.org/doc/libs/1_49_0/libs/filesystem/v3/doc/tutorial.html

#include <iostream>
#include "UseBoost.hh"


int main(int argc, char* argv[])
{
  
   if (argc < 2)
   {
      std::cout << "Usage: TestUseBoost separated path elememts to be joined into a path to file.txt\n";
      return 1;
   }

   const char* path = concat_path( argc, argv ); 

   dump_file_size(path);



   return 0;

}

