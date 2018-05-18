//  https://www.boost.org/doc/libs/1_49_0/libs/filesystem/v3/doc/tutorial.html

#include <iostream>
#include "UseBoost.hh"


int main(int argc, char* argv[])
{
  if (argc < 2)
  {
     std::cout << "Usage: tut1 path\n";
     dump_file_size(argv[0]);
     return 1;
  }

   dump_file_size(argv[1]);

  return 0;

}

