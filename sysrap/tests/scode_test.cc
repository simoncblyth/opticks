#include <iostream>
#include "scode.h"

int main()
{
    std::string code = scode::load("example","top.glsl");
    std::cout << code << "\n" ;
    return 0 ;
}
