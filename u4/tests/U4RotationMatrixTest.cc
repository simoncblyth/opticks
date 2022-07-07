
#include <vector>
#include <string>
#include "U4RotationMatrix.h"

int main(int argc, char** argv)
{
     std::vector<std::string> flips = {"null", "", "some_string",  "X", "Y", "Z", "XY", "XZ", "YZ", "XYZ", "x"  } ; 

     for(unsigned i=0 ; i < flips.size() ; i++)
     {
         const char* flip_ = flips[i].c_str(); 
         const char* flip = strcmp(flip_, "null") == 0 ? nullptr : flip_ ; 

         U4RotationMatrix* m = U4RotationMatrix::Flip(flip); 
         std::cout << flip_ << std::endl << *m << std::endl ;
     }
     return 0 ; 
}
