// name=ConstExprTest ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

/**

https://stackoverflow.com/questions/29844028/explain-constexpr-with-const-charconst


https://en.cppreference.com/w/cpp/language/constexpr
    constexpr is since C++11

    The constexpr specifier declares that it is possible to evaluate the value of
    the function or variable at compile time. Such variables and functions can then
    be used where only compile time constant expressions are allowed (provided that
    appropriate function arguments are given).


*In summary think of constexpr as more const than const : a compile time constant*   

**/


struct ConstExprTest 
{
    static constexpr const char* STR = "some useful string constant";
};


#include <iostream>

int main()
{
    std::cout << ConstExprTest::STR << std::endl ; 
    return 0 ; 
}

