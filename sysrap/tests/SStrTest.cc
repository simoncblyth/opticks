/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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

void test_StartsWith()
{
    const char* s = "/hello/there/Cathode/World" ; 
    assert( SStr::StartsWith(s, "/hello") == true ); 
    assert( SStr::StartsWith(s, "World") == false ); 
}





void test_HasPointerSuffix()
{

    std::vector<std::string> yes = 
      {
         "det0x110d9a820",
         "0x110d9a820" ,
         "0xdeadbeef0" 
      }
   ;

    std::vector<std::string> no = 
      {
         "tooshort",
         "0xdeadbeef",
         "0xdeadbeef"
      }
   ;

    for( unsigned i=0 ; i < yes.size() ; i++) 
    {
        std::cout << "y: " << yes[i] << std::endl ; 
        assert( SStr::HasPointerSuffix(yes[i].c_str(), 9) == true );
    }
    for( unsigned i=0 ; i < no.size() ; i++) 
    { 
        std::cout << "n: " << no[i] << std::endl ; 
        assert( SStr::HasPointerSuffix(no[i].c_str(), 9) == false );
    }

}


void test_HasPointerSuffix2()
{
    const char* name = "World0x7fc10641cbb0" ; 
    assert( SStr::HasPointerSuffix( name, 9, 12 ) == true ); 

    assert( SStr::GetPointerSuffixDigits("World0x7fc10641cbb0") == 12 ); 
    assert( SStr::GetPointerSuffixDigits("World0x7fc10641cbb") == 11 ); 
    assert( SStr::GetPointerSuffixDigits("World0x7fc10641cb") == 10 ); 
    assert( SStr::GetPointerSuffixDigits("World0x7fc10641c") == 9 ); 
    assert( SStr::GetPointerSuffixDigits("World0x7fc10641") == 8 ); 
    assert( SStr::GetPointerSuffixDigits("World0x7fc1064") == 7 ); 
    assert( SStr::GetPointerSuffixDigits("World0x7fc106") == 6 ); 
    assert( SStr::GetPointerSuffixDigits("World0x7fc10") == 5 ); 
    assert( SStr::GetPointerSuffixDigits("World0x7fc1") == 4 ); 
    assert( SStr::GetPointerSuffixDigits("World0x7fc") == 3 ); 
    assert( SStr::GetPointerSuffixDigits("World0x7f") == 2 ); 
    assert( SStr::GetPointerSuffixDigits("World0x7") == 1 ); 
    assert( SStr::GetPointerSuffixDigits("World0x") == 0 ); 
    assert( SStr::GetPointerSuffixDigits("World0") == -1 ); 
    assert( SStr::GetPointerSuffixDigits("World") == -1 ); 
    assert( SStr::GetPointerSuffixDigits("") == -1 ); 
    assert( SStr::GetPointerSuffixDigits(NULL) == -1 ); 
}


void test_Replace()
{
    const char* name = "TITAN RTX " ; 
    const char* xname2 = "TITAN_RTX_" ; 
    const char* name2 = SStr::Replace(name, ' ', '_' ); 
    assert( strcmp(name2, xname2) == 0 );  
}

void test_ReplaceEnd()
{
    const char* name = "/some/path/to/hello.ppm" ; 
    const char* xname2 = "/some/path/to/hello.npy" ; 
    const char* name2 = SStr::ReplaceEnd(name, ".ppm", ".npy" ); 
    assert( strcmp(name2, xname2) == 0 );  
}

void test_ArrayToString()
{
    // thinking about optix7c- and embedded_ptx_code from bin2c 
    // observe that without NULL termination get garbage on the end of the string 
    // which is why must use "--padd 0" which sets trailing bytes::
    // 
    //       bin2c --name data_variable_name --padd 0 inputfile > data.c 
    //
    const char imageBytes[] = { 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x00 } ; 
    std::string s = imageBytes ; 
    std::cout << "[" << s << "]" << std::endl ; 
    assert( s.compare("ABCDEF") == 0 ); 
    assert( 'A' == 0x41 ); 
}





void test_Save()
{
    std::vector<std::string> v = { "red", "green", "blue", "cyan", "magenta", "yellow", "green" } ; 
    const char* path = "$TMP/SStrTest_test_Save.txt" ; 
    SStr::Save(path, v ); 
}




int main(int argc , char** argv )
{
    OPTICKS_LOG(argc, argv);

    /*
    test_ToULL();
    test_FromULL();
    test_Format1();  
    test_Contains();  
    test_EndsWith();  
    test_HasPointerSuffix();  
    test_HasPointerSuffix2();  
    test_StartsWith();  
    test_Replace();  
    test_ReplaceEnd();  
    test_ArrayToString();  
    */
    test_Save();  

    return 0  ; 
}
// om-;TEST=SStrTest om-t
