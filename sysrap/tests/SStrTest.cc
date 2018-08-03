// om-;TEST=SStrTest om-t 

#include <cassert>
#include <string>
#include "SStr.hh"

#include "OPTICKS_LOG.hh"


void test_ToULL()
{
    char* s = new char[8+1] ; 
    s[0] = '\1' ; 
    s[1] = '\2' ; 
    s[2] = '\3' ; 
    s[3] = '\4' ; 
    s[4] = '\5' ; 
    s[5] = '\6' ; 
    s[6] = '\7' ; 
    s[7] = '\7' ; 
    s[8] = '\0' ; 
    
    typedef unsigned long long ULL ; 

    ULL v = SStr::ToULL(s ); 

    LOG(info) << " v " << std::hex << v ;

    assert( 0x707060504030201ull == v );
}

void test_FromULL()
{
    typedef unsigned long long ULL ; 
    const char* s0 = "0123456789" ; 
    ULL v = SStr::ToULL(s0); 

    const char* s1 = SStr::FromULL( v ); 
    LOG(info) 
        << " s0 " << std::setw(16) << s0 
        << " s1 " << std::setw(16) << s1   
        ;

    ULL v0 = SStr::ToULL(NULL) ; 
    assert( v0 == 0ull ); 

}




void test_Format1()
{
    const char* fmt = "hello %s hello"  ; 
    const char* value = "world" ; 
    const char* result = SStr::Format1<256>(fmt, value );
    const char* expect = "hello world hello" ; 
    assert( strcmp( result, expect) == 0 ); 

    // this asserts from truncation 
    //const char* result2 = SStr::Format1<16>(fmt, value );
    //LOG(info) << " result2 " << result2 ;  
 
}

void test_Contains()
{
    const char* s = "/hello/there/Cathode/World" ; 
    assert( SStr::Contains(s, "Cathode") == true ); 
    assert( SStr::Contains(s, "cathode") == false ); 
}
void test_EndsWith()
{
    const char* s = "/hello/there/Cathode/World" ; 
    assert( SStr::EndsWith(s, "Cathode") == false ); 
    assert( SStr::EndsWith(s, "World") == true ); 
}


int main(int argc , char** argv )
{
    OPTICKS_LOG(argc, argv);

    /*
    test_ToULL();
    test_FromULL();
    test_Format1();  
    test_Contains();  
    */
    test_EndsWith();  

    return 0  ; 
}

