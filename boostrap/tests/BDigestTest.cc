#include "BDigest.hh"
#include <cstring>
#include <cassert>
#include <iostream>

int main(int, char** argv)
{
    const char* str = "hello" ;
    unsigned int len = strlen(str);
    void* buf = (void*)str ; 

    std::string a = BDigest::md5digest(str, len );
    std::string b = BDigest::digest(buf, len );
    assert( a.compare(b) == 0 );
    std::cerr << argv[0] 
              << " " 
              << buf
              << " --> "
              << a 
              ;

    return 0 ; 
}
