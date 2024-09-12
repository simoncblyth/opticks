
#include <iostream>

int main()
{

#if __cplusplus == 201103
    std::cout << " detected C++11\n" ; 
#elif __cplusplus == 201402
    std::cout << " detected C++14\n" ; 
#elif __cplusplus == 201703
    std::cout << " detected C++17\n" ; 
#elif __cplusplus == 202002
    std::cout << " detected C++20\n" ; 
#else
    std::cout << " detected UNEXPECTED C++ \n" ; 
#endif



    long cpp = __cplusplus ; 
    int i_cpp = int(cpp); 
    int u_cpp = int((unsigned char)(cpp)); 
    std::cout << cpp << " " << i_cpp << " " << u_cpp << std::endl ;
    return u_cpp ; 
}
