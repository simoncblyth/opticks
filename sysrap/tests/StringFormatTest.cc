// name=StringFormatTest ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <iostream>
#include <cstdio>
#include <cassert>
#include <vector>
#include <string>

template<typename ... Args>
std::string Format( const char* fmt, Args ... args )
{
    int sz = std::snprintf( nullptr, 0, fmt, args ... ) + 1; // Extra space for '\0'
    assert( sz > 0 );  
    std::vector<char> buf(sz) ;   
    std::snprintf( buf.data(), sz, fmt, args ... );
    return std::string( buf.begin(), buf.begin() + sz - 1 );  // exclude null termination 
}

int main(int argc, char** argv)
{
    const char* fmt = " Hello %d World %4.2f \n" ; 
    std::string s = Format(fmt, 101, 50.5f ); 
    std::cout << "[" << s << "]" << std::endl ; 
    return 0 ; 
}


