#include <string>
#include <cstring>
#include <cassert>
#include <iostream>

#include "SSys.hh"
#include "SDigest.hh"
#include "OPTICKS_LOG.hh"

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


void test_digest_vec()
{
   std::vector<std::string> v ; 
   v.push_back("red");
   v.push_back("green");
   v.push_back("blue");

   std::string dig0 = SDigest::digest(v); 


   v.push_back("blue");
   v.push_back("blue");
   v.push_back("blue");
   v.push_back("blue");
   v.push_back("blue");

   std::string dig1 = SDigest::digest(v); 
   std::string dig2 = SDigest::digest_skipdupe(v); 

   LOG(info) 
        << " dig0 " << dig0
        << " dig1 " << dig1
        << " dig2 " << dig2
        ;


   assert( dig0.compare(dig1.c_str()) != 0 );
   assert( dig0.compare(dig2.c_str()) == 0 );


}

void test_IsDigest()
{
    assert( SDigest::IsDigest(NULL) == false );
    assert( SDigest::IsDigest("0123") == false );
    assert( SDigest::IsDigest("0123456789abcdef") == false );
    assert( SDigest::IsDigest("0123456789abcdef0123456789abcdef") == true );
}

void test_DigestPath(const char* path)
{
    std::string d0 = SDigest::DigestPath(path) ; 
    std::string d1 = SDigest::DigestPath2(path) ; 
    assert( d0.compare(d1) == 0 ); 

    if(SSys::getenvint("VERBOSE",0) == 1)
    { 
        std::cout << "SDigest::DigestPath  "  << d0 << std::endl ; 
        std::cout << "SDigest::DigestPath2 "  << d1 << std::endl ; 
    }
    std::cout << d0 << std::endl ;
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //test_static();
    //test_update();

    //test_digest_vec();
    //test_IsDigest();

    const char* path = argc > 1 ? argv[1] : argv[0] ; 
    test_DigestPath(path);

    return 0 ; 
}

