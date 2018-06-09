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




int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    //test_static();
    //test_update();

    //test_digest_vec();

    test_IsDigest();

    return 0 ; 
}

