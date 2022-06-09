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

**/

#include <cassert>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

#include "sdigest.h"
#include "ssys.h"

int main(int argc, char** argv)
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

    std::vector<std::string> dig(4) ; 
    dig[0] = sdigest::buf( dat, 5 ); 
    dig[1] = sdigest::buf( m , strlen(m) ); 
    dig[2] = sdigest::buf( msg.c_str(), strlen(msg.c_str()) ); 
    dig[3] = ssys::popen( cmd.c_str() ); 

    for(unsigned i=0 ; i < dig.size() ; i++) 
    {
        std::cout 
            << " i " << std::setw(3) << i 
            << " : " << dig[i] 
            << std::endl
            ;
        assert( strcmp( dig[0].c_str(), dig[i].c_str() ) == 0 );  
    }
    return 0 ;  
}
