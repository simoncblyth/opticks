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


::

    In [6]: import hashlib
    In [7]: hashlib.md5(b"hello").hexdigest()
    Out[7]: '5d41402abc4b2a76b9719d911017c592'

    In [12]: a = np.array(b"hello", dtype="|S5" )
    In [13]: a.data
    Out[13]: <memory at 0x16ab67d60>
    In [14]: hashlib.md5(a.data).hexdigest()
    Out[14]: '5d41402abc4b2a76b9719d911017c592'

    In [15]: a2 = np.array([b"hello", b"world"], dtype="|S5" )
    In [16]: a2 
    Out[16]: array([b'hello', b'world'], dtype='|S5')
    In [17]: a2[0].data
    Out[17]: <memory at 0x16aaddef0>
    In [18]: hashlib.md5(a2[0].data).hexdigest()
    Out[18]: '5d41402abc4b2a76b9719d911017c592'

**/

#include <cassert>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

#include "sdigest.h"
#include "ssys.h"
#include "sstr.h"

void check(const std::vector<std::string>& dig, const char* known=nullptr )
{
    for(unsigned i=0 ; i < dig.size() ; i++) 
    {
        std::cout 
            << " i " << std::setw(3) << i 
            << " : " << dig[i] 
            << std::endl
            ;
        assert( strcmp( dig[0].c_str(), dig[i].c_str() ) == 0 );  
        if(known) assert( strcmp(known, dig[i].c_str() ) == 0 );
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
    const char* hello_digest = "5d41402abc4b2a76b9719d911017c592" ; 

    std::string msg = "hello" ; 

    std::stringstream ss ; 
#ifdef __APPLE__
    ss << "md5 -q -s " << m ; 
#else
    ss << "echo " << hello_digest ;   
    // kludge as Linux equivalent "echo -n hello | md5sum" needs a pipe  
#endif
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

    check(dig, hello_digest ); 
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

void test_Desc()
{
    std::cout << "sdigest::Desc() " << sdigest::Desc() << std::endl ; 
}

int main(int argc, char** argv)
{
    /*
    test_int(); 
    */
    test_hello(); 
    test_Desc(); 

    return 0 ;  
}
