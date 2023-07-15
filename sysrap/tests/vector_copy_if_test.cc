// name=vector_copy_if_test ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>

int main()
{
    std::vector<int> all ;
 
    std::vector<int> a = {0, 1, -2, 3, -10, -20, -30 };
    std::vector<int> b = {0, 100, 200, 300, -10, -20, -30 };
    std::vector<int> c = {0, -1, 20, 30, -10, -20, -30 };

    auto predicate = [](int i){return i>0;} ; 

    std::copy_if( a.begin(), a.end(), std::back_inserter(all), predicate ) ; 
    std::copy_if( b.begin(), b.end(), std::back_inserter(all), predicate ) ; 
    std::copy_if( c.begin(), c.end(), std::back_inserter(all), predicate ) ; 

    std::cout << " all.size " << all.size() << std::endl ; 

    std::cout << " all[ " ; 
    for(int i=0 ; i < int(all.size()); i++) std::cout << all[i] << " " ; 
    std::cout << "]" << std::endl ;     

    return 0 ; 
}
  


