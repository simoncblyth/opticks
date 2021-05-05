// name=SMacroStringify ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <iostream>

#define foo 4
#define bar hello


// https://gcc.gnu.org/onlinedocs/gcc-4.8.5/cpp/Stringification.html
#define xstr(s) str(s)
#define str(s) #s

int main(int argc, char** argv)
{
    std::cout << "foo:" << xstr(foo) << std::endl ; 
    std::cout << "bar:" << xstr(bar) << std::endl ; 
    return 0 ; 
}

