// ~/o/sysrap/tests/vector_clear_test.sh

#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>

void dump(const std::vector<int>& all, size_t num, const char* msg)
{
    std::cout << " all.size " << all.size() << " num " << num ;
    std::cout << " [ " ;
    for(size_t i=0 ; i < num ; i++) std::cout << all[i] << " " ;
    std::cout << "] -- " << msg << "\n"  ;
}

int main(int argc, char** argv)
{

    std::cout << argv[0] << " -- Compiled with gcc " <<  __VERSION__ << "\n" ;
    size_t num = 4 ;
    std::vector<int> all(num) ;
    dump(all, num, "before deliberate clear bug");

    all.clear();  // investigate crash OR not-crash in various gcc Debug/Release config
    all[0] = 100 ;
    all[1] = 200 ;
    all[2] = 300 ;
    all[3] = 400 ;

    dump(all, num, "after clear and undefined [] filling");


    return 0 ;
}



