// ./smeta_test.sh

#include "smeta.h"

int main()
{
    std::string meta ; 
    smeta::Collect(meta, "tests/smeta_test.cc"); 
    std::cout << meta << std::endl ; 
    return 0 ; 
}
