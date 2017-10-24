#include <string>
#include <cstring>
#include <cassert>
#include <iostream>

#include "SDigest.hh"
#include "PLOG.hh"

void test_static()
{
    const char* str = "hello" ;
    unsigned int len = strlen(str);
    void* buf = (void*)str ; 

    std::string a = SDigest::md5digest(str, len );
    std::string b = SDigest::digest(buf, len );
    assert( a.compare(b) == 0 );

    LOG(info) << str
              << " --> "
              << a 
              ;
}

void test_update()
{
   const char* str = "hello" ;
   SDigest dig ; 
   dig.update( const_cast<char*>(str), sizeof(char)*strlen(str) );
   char* dgst = dig.finalize();

   LOG(info) << str 
             << " --> "
             << dgst 
             ;
}

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    test_static();
    test_update();

    return 0 ; 
}

