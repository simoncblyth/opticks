// name=sdigest_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name
/**
sdigest_test.cc
=================

When comparing with digests from files beware of the newline::

    epsilon:tests blyth$ printf "hello" > /tmp/hello  # echo includes newline 
    epsilon:tests blyth$ cat /tmp/hello 
    helloepsilon:tests blyth$ 

    epsilon:tests blyth$ md5 /tmp/hello
    MD5 (/tmp/hello) = 5d41402abc4b2a76b9719d911017c592
    epsilon:tests blyth$ 

    epsilon:tests blyth$ md5 -q -s hello
    5d41402abc4b2a76b9719d911017c592

::

    epsilon:tests blyth$ name=sdigest_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name
     i   0 : 5d41402abc4b2a76b9719d911017c592
     i   1 : 5d41402abc4b2a76b9719d911017c592
     i   2 : 5d41402abc4b2a76b9719d911017c592
     i   3 : 5d41402abc4b2a76b9719d911017c592


**/

#include <cassert>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

#include "sdigest.h"
#include "ssys.h"
#include "sstr.h"

void check(const std::vector<std::string>& dig )
{
    for(unsigned i=0 ; i < dig.size() ; i++) 
    {
        std::cout 
            << " i " << std::setw(3) << i 
            << " : " << dig[i] 
            << std::endl
            ;
        assert( strcmp( dig[0].c_str(), dig[i].c_str() ) == 0 );  
    }
}


void test_hello()
{
    char dat[6] ; 
    dat[0] = 'h' ; 
    dat[1] = 'e' ; 
    dat[2] = 'l' ; 
    dat[3] = 'l' ; 
    dat[4] = 'o' ; 
    dat[5] = '\0' ; 

    const char* m = "hello" ; 
    std::string msg = "hello" ; 

    std::stringstream ss ; 
    ss << "md5 -q -s " << m ; 
    std::string cmd = ss.str(); 

    std::vector<std::string> dig(9) ; 

    dig[0] = sdigest::Buf( dat, 5 );         // null terminator not included 
    dig[1] = sdigest::Buf( m , strlen(m) );  // strlen does not count terminator 
    dig[2] = sdigest::Buf( msg.c_str(), strlen(msg.c_str()) ); 
    dig[3] = ssys::popen( cmd.c_str() ); 


    sdigest u0, u1, u2, u3 ;

    u0.add(m);   
    u1.add(msg);   
    u2.add(dat, 5);   
    for(int i=0 ; i < 5 ; i++) u3.add(dat+i, 1 );  // adding character by character

    dig[4] = u0.finalize();  
    dig[5] = u1.finalize();  
    dig[6] = u2.finalize();  
    dig[7] = u3.finalize();  

    const char* path = "/tmp/hello.txt" ; 
    sstr::Write(path, m ); 
    dig[8] = sdigest::Path(path) ; 

    check(dig); 
}

void test_int()
{
    sdigest u0 ; 
    u0.add(0); 
    u0.add(1); 
    u0.add(2); 
    u0.add(3); 

    std::vector<int> ii = {{ 0,1,2,3 }} ; 
    sdigest u1 ; 
    u1.add( (char*)ii.data(), sizeof(int)*ii.size() ); 


    std::vector<std::string> dig(2) ; 
    dig[0] = u0.finalize() ; 
    dig[1] = u1.finalize() ; 

    check(dig); 
}



int main(int argc, char** argv)
{
    /*
    test_int(); 
    */
    test_hello(); 
    return 0 ;  
}
