#include <string>
#include <cstring>
#include <cassert>
#include <iostream>
#include <fstream>

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



void test_DigestPathInByteRange(const char* path, int i0, int i1)
{
    LOG(info) 
        << " path " << path 
        << " i0 " << i0
        << " i1 " << i1
        ;
        
    std::string d0 = SDigest::DigestPathInByteRange(path, i0, i1 ) ; 
    std::cout << d0 << std::endl ;
}

void test_DigestPathInByteRange()
{
    unsigned n = 100 ;  
    const char* path = "/tmp/test_DigestPathInByteRange.txt" ; 
    const char* a = "0123456789abcdef" ; 
    unsigned bufsize = 10 ; 


    std::ofstream out(path, std::ios::out);
    for(unsigned i=0 ; i < n ; i++ ) out << a ; 
    out.close(); 

    LOG(info) 
        << " path " << path 
        << " bufsize " << bufsize 
        ; 

    std::string d0, d1 ; 

    for(unsigned i=0 ; i < n ; i++)
    { 
        int i0 = i*strlen(a) ; 
        int i1 = (i+1)*strlen(a) ; 

        if(d0.empty()) 
        {
            d0 = SDigest::DigestPathInByteRange(path, i0, i1, bufsize ) ; 
            continue ; 
        }
             
        d1 = SDigest::DigestPathInByteRange(path, i0, i1, bufsize ) ;

        bool match = d0.compare(d1) == 0 ;  


        LOG(info) 
            << " i "  << std::setw(5) << i  
            << " i0 "  << std::setw(5) << i0  
            << " i1 "  << std::setw(5) << i1  
            << " d0 " << d0 
            << " d1 " << d1
            << ( !match ? " MISMATCH " : "" ) 
            ; 

        assert( match ); 
    }

}


void test_main( int argc, char** argv )
{
    const char* path = argc > 1 ? argv[1] : argv[0] ; 
    int i0 = argc > 2 ? atoi(argv[2]) : -1 ; 
    int i1 = argc > 3 ? atoi(argv[3]) : -1 ; 

    if( i0 > -1 && i1 > -1 ) 
    {
        test_DigestPathInByteRange(path, i0, i1);
    } 
    else
    {
        test_DigestPath(path);
    }
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //test_static();
    //test_update();

    //test_digest_vec();
    //test_IsDigest();


    test_main(argc, argv);

    //test_DigestPathInByteRange(); 


    return 0 ; 
}

