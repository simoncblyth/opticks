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
#include <csignal>
#include <string>
#include <iostream>
#include <iomanip>

#include "SStr.hh"
#include "SPath.hh"

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
    bool v0_expect = v0 == 0ull ;
    assert( v0_expect); 
    if(!v0_expect) std::raise(SIGINT); 

}




void test_Format1()
{
    const char* fmt = "hello %s hello"  ; 
    const char* value = "world" ; 
    const char* result = SStr::Format1<256>(fmt, value );
    const char* expect = "hello world hello" ; 

    bool result_expect = strcmp( result, expect) == 0 ;
    assert(result_expect ); 
    if(!result_expect) std::raise(SIGINT); 

    // this asserts from truncation 
    //const char* result2 = SStr::Format1<16>(fmt, value );
    //LOG(info) << " result2 " << result2 ;  
 
}


void test_FormatInt()
{
    const char* fmt = "/tmp/Frame%0.3d.ppm"   ; 
    LOG(info) << fmt ; 
    for(int i=-10 ; i < 10 ; i++ )
    {
        const char* result = SStr::FormatInt<64>(fmt, i );
        std::cout << result << std::endl ;   
    }
}





void test_FormatInt_2()
{
    const char* fmt = "%d"   ; 
    LOG(info) << fmt ; 
    for(int i=-10 ; i < 10 ; i++ )
    {
        const char* result = SStr::FormatInt<8>(fmt, i );
        std::cout << result << std::endl ;   
    }
}


void test_FormatIndex()
{
    for(int i=-10 ; i <= 10 ; i++ )
    {
        const char* result = SStr::FormatIndex(i);
        std::cout << " i " << std::setw(4) << i << "[" << result << "]" << std::endl ;   
    }
}




void test_Contains()
{
    const char* s = "/hello/there/Cathode/World" ; 

    bool expect = SStr::Contains(s, "Cathode") == true && SStr::Contains(s, "cathode") == false  ;
    assert( expect ); 
    if(!expect) std::raise(SIGINT); 
}
void test_EndsWith()
{
    const char* s = "/hello/there/Cathode/World" ; 
    bool expect = SStr::EndsWith(s, "Cathode") == false && SStr::EndsWith(s, "World") == true  ;
    assert( expect ); 
    if(!expect) std::raise(SIGINT); 
}

void test_StartsWith()
{
    const char* s = "/hello/there/Cathode/World" ; 
    bool expect = SStr::StartsWith(s, "/hello") == true && SStr::StartsWith(s, "World") == false ;
    assert( expect ); 
    if(!expect) std::raise(SIGINT); 
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
    bool expect = SStr::HasPointerSuffix( name, 9, 12 ) == true ;
    assert(expect) ; 
    if(!expect) std::raise(SIGINT) ; 

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

    bool expect = strcmp(name2, xname2) == 0 ;
    assert(expect);  
    if(!expect) std::raise(SIGINT) ; 
}

void test_ReplaceEnd()
{
    const char* name = "/some/path/to/hello.ppm" ; 
    const char* xname2 = "/some/path/to/hello.npy" ; 
    const char* name2 = SStr::ReplaceEnd(name, ".ppm", ".npy" ); 
    bool expect = strcmp(name2, xname2) == 0 ;
    assert(expect);  
    if(!expect) std::raise(SIGINT) ; 
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


void test_LoadList()
{
    std::vector<std::string> v0 = { "red", "green", "blue", "cyan", "magenta", "yellow", "green" } ; 
    
    const char* path = SPath::Resolve("$TMP/SStrTest/test_LoadList.txt", FILEPATH) ; 
    SStr::Save(path, v0 ); 

    std::vector<std::string> v1 ;    
    SStr::LoadList(path, v1 ); 
    assert( v0.size() == v1.size() ); 


    std::vector<std::string>* v2p = SStr::LoadList(path);  
    assert( v2p ); 
    const std::vector<std::string>& v2 = *v2p ; 

    assert( v2.size() == v0.size() ); 


    for( unsigned i=0 ; i < v0.size() ; i++) 
         std::cout 
             << std::setw(20) << v0[i] 
             << " : " 
             << std::setw(20) << v1[i] 
             << " : " 
             << std::setw(20) << v2[i] 
             << " : " 
             << std::endl
             ; 
}









const char* TXT = R"LITERAL(
red
green
blue
cyan
magenta
yellow
)LITERAL" ; 

void test_Save_Load()
{
    const char* path = "$TMP/SStrTest/test_Save_Load.txt" ; 
    SStr::Save(path, TXT );

    const char* txt = SStr::Load(path); 

    LOG(info) << " TXT [" << TXT << "]"  ; 
    LOG(info) << " txt [" << txt << "]"  ; 

    assert( strcmp(txt, TXT) == 0 );  
}

void test_Save_PWD()
{
    const char* path = "test_Save_PWD.txt" ; 
    SStr::Save(path, TXT );  
}




void test_Split()
{
    std::vector<std::string> elem ; 
    const char* str = "red,green,blue,cyan,magenta,yellow" ; 
    SStr::Split(str, ',', elem ); 
    assert( elem.size() == 6 ); 
    for(int i=0 ; i < int(elem.size()) ; i++) std::cout << elem[i] << std::endl ; 
}


void test_Concat_()
{
    std::cout << SStr::Concat_("hello/", 1, ".npy" ) << std::endl ; 
}


void test_AsInt()
{
    const char* arg = "00000" ; 
    int i = SStr::AsInt(arg); 
    bool i_expect = i == 0 ;
    assert( i_expect ); 
    if(!i_expect) std::raise(SIGINT); 
}


void test_ExtractInt()
{
    const char* path = "/some/long/path/with_00000.jpg" ; 
    int i = SStr::ExtractInt(path, -9, 5 ); 
    std::cout << "path " << path << " i " << i << std::endl ;  
    assert( i == 0 ); 
}

void test_SimpleMatch_WildMatch()
{
    std::vector<std::string> labels = {
        "r0",
        "r1",
        "r2",
        "r3",
        "r1p0","r1p1","r1p2","r1p3",
        "r2p0","r2p1","r2p2","r2p3",
        "R3P0N0",
        "R3P0N1",
        "R3P0N2",
        "R3P1N0",
        "R3P1N1",
        "R3P1N2",
        "R3P1N3",
    }; 
   
    std::vector<std::string> querys = { 
         "r2", 
         "r2$", 
         "r2p", 
         "r2p$", 
         "r2p2$", 
         "R3P1", 
         "R3P1*", 
         "R3P1N", 
         "R3P1N?", 
         "R3P1N2$", 
         "R3P?N0", 
         "R3P1*", 
      } ; 

    for(int i=0 ; i < int(querys.size()) ; i++)
    {
        const char* q = querys[i].c_str() ;   
        unsigned lq = strlen(q); 
        bool qed = q[lq-1] == '$' ;

        std::cout 
            << " q " << q 
            << " lq " << lq 
            << " qed: " << ( qed ? "Y" : "N" )
            << std::endl 
            ;


        for(int j=0 ; j < int(labels.size()) ; j++)
        {
            const char* s = labels[j].c_str(); 
            bool sm = SStr::SimpleMatch(s,q); 
            bool wm = SStr::Match(s,q); 
            std::cout 
                << " SStr::SimpleMatch(" 
                << std::setw(7) << s 
                << " , "
                << std::setw(7) << q 
                << " )  : "
                << ( sm ? "Y" : " " )
                ;

            std::cout 
                << " SStr::Match(" 
                << std::setw(7) << s 
                << " , "
                << std::setw(7) << q 
                << " )  : "
                << ( wm ? "Y" : " " )
                ;

           std::cout << std::endl ; 
           



        }
    }
}


void test_ISplit()
{
    LOG(info); 

    {
        const char* wavelength = "380,400,420,440,460" ; 
        std::vector<int> inm ; 
        SStr::ISplit(wavelength, inm, ',' ); 
        assert( inm.size() == 5 ); 
        assert( inm[0] == 380 ); 
        assert( inm[1] == 400 ); 
        assert( inm[2] == 420 ); 
        assert( inm[3] == 440 ); 
        assert( inm[4] == 460 ); 
    }
    {
        const char* wavelength = "0" ; 
        std::vector<int> inm ; 
        SStr::ISplit(wavelength, inm, ',' ); 
        assert( inm.size() == 1 ); 
        assert( inm[0] == 0 ); 
    }
    {
        const char* wavelength = "440" ; 
        std::vector<int> inm ; 
        SStr::ISplit(wavelength, inm, ',' ); 
        assert( inm.size() == 1 ); 
        assert( inm[0] == 440 ); 
    }

}


void test_FormatReal()
{
    double value = 1.1 ; 
    const char* s = SStr::FormatReal<double>(value, 6, 4, '0'); 

    std::cout 
        << " value " << value 
        << " s [" << s  << "]" 
        << std::endl 
        ;
}

void test_StripPrefix()
{
    const char* lines = R"LITERAL(
/dd/Materials/red
/dd/Materials/green
/dd/Materials/blue
_dd_Materials_red
_dd_Materials_green
_dd_Materials_blue
red
green
blue
)LITERAL" ; 

    std::stringstream ss(lines) ;    
    std::string line ; 
    while (std::getline(ss, line))  
    {   
        if(line.empty()) continue ;   

        const char* s = line.c_str(); 
        const char* sp = SStr::StripPrefix(s, "/dd/Materials/", "_dd_Materials_" );          
        const char* sp2 = SStr::MaterialBaseName(s); 

        bool sp_expect = strcmp( sp, sp2 ) == 0  ;
        assert(sp_expect); 
        if(!sp_expect) std::raise(SIGINT); 

        std::cout 
            << std::setw(50) << line 
            << " : "
            << std::setw(50) << sp
            << std::endl
            ;
    }    
}


void test_TrimPointerSuffix()
{
    const char* lines = R"LITERAL(
Hello0xdeadbeef
Hello0xnope
Hello0xnope1
Hello0xnope1
Hello0x0123
Hello0x01234
Hello0x012345
Hello0x0123456
Hello0x01234567
Hello0x012345678
Hello0x0123456789
Hello0x0123456789a
Hello0xa
Hello0xab
Hello0xabc
Hello0xabcd
Hello0xabcde
Hello0xabcdef
Hello0xabcdef0
Hello0xabcdef01
Hello0xabcdef012
0xcafecafe
0xdeadbeef
a0xcafecafe
a0xdeadbeef
)LITERAL" ; 

    // the suffix chars must be valid hexdigits

    std::stringstream ss(lines) ;    
    std::string line ; 
    while (std::getline(ss, line))  
    {   
        if(line.empty()) continue ;   

        const char* s = line.c_str(); 
        const char* sp = SStr::TrimPointerSuffix(s); 
        std::cout 
            << std::setw(50) << s
            << " : "
            << std::setw(50) << sp
            << std::endl
            ;
    }    
}


void test_ReplaceChars()
{
     const char* str0 = "(-0.585,-0.805, 0.098, 0.000) (-0.809, 0.588, 0.000, 0.000) (-0.057,-0.079,-0.995, 0.000) (1022.116,1406.822,17734.953, 1.000)"  ; 
     const char* str1 = SStr::ReplaceChars(str0); 

     std::cout 
         << " str0 " << str0 << std::endl 
         << " str1 " << str1 << std::endl 
         ;
}

void test_ato_()
{
    const char* a = "104.25" ; 

    float f = SStr::ato_<float>(a);  
    double d = SStr::ato_<double>(a); 
    int i = SStr::ato_<int>(a);  
    unsigned u = SStr::ato_<unsigned>(a);  


    LOG(info) 
       << " a " << a 
       << " f " << std::setw(10) << std::fixed << std::setprecision(4) << f 
       << " d " << std::setw(10) << std::fixed << std::setprecision(4) << d
       << " i " << std::setw(10) << i 
       << " u " << std::setw(10) << u 
       ; 
}

void test_Extract()
{
    const char* s = "asjdhajsdhas-100   -200 300 sajdasjdhakjHDKJ +66 21 23 45 1001 -10 akjdshaHD -42 " ; 
    LOG(info) << s ; 
    std::vector<long> vals ; 
    SStr::Extract_(vals, s ); 

    for(unsigned i=0 ; i < vals.size() ; i++ ) std::cout << vals[i] << std::endl;

}

void test_Extract_float()
{
    const char* s = "asjdhajsdhas-0.1   -.2 +30.5 sajdasjdhakjHDKJ +66 21 23.6 45 1001 -10.2 akjdshaHD -42.5 " ; 
    LOG(info) << s ; 
    std::vector<float> vals ; 
    SStr::Extract_(vals, s ); 

    for(unsigned i=0 ; i < vals.size() ; i++ ) std::cout << vals[i] << std::endl;
}

void test_Trim(const char* s)
{
     std::cout 
         << "s                    [" << s << "]" << std::endl 
         << "SStr::TrimTrailing(s)[" << SStr::TrimTrailing(s) << "]" << std::endl 
         << "SStr::TrimLeading(s) [" << SStr::TrimLeading(s) << "]" << std::endl 
         << "SStr::Trim(s)        [" << SStr::Trim(s) << "]" << std::endl 
         ;

}


void test_Trim()
{
     const char* s0 = "            contents with gaps before whitespace          " ; 
     test_Trim(s0);

     const char* s1 = R"LITERAL(     
 
red
green
blue cyan magenta
yellow


)LITERAL" ;

     test_Trim(s1);


}

void test_Count()
{
    assert( SStr::Count("a bcdefg", ' ') == 1 ); 
    assert( SStr::Count("a  bcdefg", ' ') == 2 ); 
    assert( SStr::Count(" ", ' ') == 1 ); 
    assert( SStr::Count("  ", ' ') == 2 ); 
    assert( SStr::Count("", ' ') == 0 ); 
}
void test_All()
{
    assert( SStr::All("aaaaa", 'a') == true ); 
    assert( SStr::All("aabaa", 'a') == false ); 
    assert( SStr::All("", 'a') == false ); 
    assert( SStr::All(" ", ' ') == true ); 
    assert( SStr::All("  ", ' ') == true ); 
}
void test_Blank()
{
    assert( SStr::Blank("aaaaa") == false ); 
    assert( SStr::Blank("") == true ); 
    assert( SStr::Blank(" ") == true ); 
    assert( SStr::Blank("  ") == true ); 
    assert( SStr::Blank("           ") == true ); 
    assert( SStr::Blank("\n") == false ); 
}





void test_ExtractLong()
{

     const char* lines = R"LITERAL(     
 
red1
green2
blue2 cyan magenta
yellow3


)LITERAL" ;


    std::stringstream ss(lines) ;    
    std::string line ; 
    while (std::getline(ss, line))  
    {   
        if(line.empty()) continue ;   

        const char* s = line.c_str(); 
        long l = SStr::ExtractLong(s, 0); 

        LOG(info) << std::setw(40) << s << " l: " << l  ; 
    }
}


void test_HeadFirst_HeadLast()
{
    const char* lines = R"LITERAL(
SomeName_suffix
green2WithoutSuffix
MultipleUnderscoreblue2_cyan_magenta
)LITERAL" ;

    std::stringstream ss(lines) ;    
    std::string line ; 
    while (std::getline(ss, line))  
    {   
        if(line.empty()) continue ;   

        const char* s = line.c_str(); 
        const char* f = SStr::HeadFirst(s, '_'); 
        const char* l = SStr::HeadLast(s, '_'); 
       
        std::cout 
            << " s[" << std::setw(40) << ( s ? s : "-" ) << "]" << std::setw(3) << strlen(s) 
            << " f[" << std::setw(40) << ( f ? f : "-" ) << "]" << std::setw(3) << strlen(f) 
            << " l[" << std::setw(40) << ( l ? l : "-" ) << "]" << std::setw(3) << strlen(l) 
            << std::endl 
            ;
    }
}   


void test_Format_Ellipsis()
{
    LOG(info) << SStr::Format_("Hello %d World %10.4f", 101, 50.5 ); 
    LOG(info) << SStr::Format("Hello %d World %10.4f", 101, 50.5 ); 

    for(int i=0 ; i < 1000 ; i+= 100 ) 
    {
         std::cout 
            << " before " 
            << std::setw(7) << SStr::Format("key:%d", i ) 
            << " after " 
            << std::endl 
            ; 
    }

    
}


void test_StartsWithLetterAZaz()
{
    assert( SStr::StartsWithLetterAZaz(nullptr) == false ); 
    assert( SStr::StartsWithLetterAZaz("") == false ); 
    assert( SStr::StartsWithLetterAZaz(" ") == false ); 
    assert( SStr::StartsWithLetterAZaz("0") == false ); 
    assert( SStr::StartsWithLetterAZaz("0a") == false ); 
    assert( SStr::StartsWithLetterAZaz(" a") == false ); 

    assert( SStr::StartsWithLetterAZaz("a") == true ); 
    assert( SStr::StartsWithLetterAZaz("abcd") == true ); 
    assert( SStr::StartsWithLetterAZaz("Abcd") == true ); 
}


void test_ParseStringIntInt()
{
    const char* x0 = "Hello" ; 
    int y0 = 10 ; 
    int z0 = 1000 ;  
    std::stringstream ss ; 
    ss << x0 << ":" << y0 << ":" << z0 ; 

    std::string s = ss.str(); 
    const char* triplet = s.c_str(); 

    int y1 = 0 ; 
    int z1 = 0 ; 
    const char* x1 = SStr::ParseStringIntInt(triplet, y1, z1); 

    bool x_expect =  strcmp(x0,x1) == 0 ;
    bool y_expect = y0 == y1 ;
    bool z_expect = z0 == z1 ;

    assert( x_expect );
    assert( y_expect );
    assert( z_expect );

    if(!x_expect) std::raise(SIGINT); 
    if(!y_expect) std::raise(SIGINT); 
    if(!z_expect) std::raise(SIGINT); 

    LOG(info); 
}



// om- ; TEST=SStrTest om-t


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
    test_Save();  
    test_Split();  
    test_Concat_(); 
    test_AsInt(); 
    test_ExtractInt(); 
    test_SimpleMatch_WildMatch(); 
    test_ISplit(); 
    test_FormatReal(); 
    test_StripPrefix(); 
    test_TrimPointerSuffix(); 
    test_ReplaceChars(); 
    test_ato_(); 
    test_Save_Load(); 
    test_Save_PWD(); 
    test_Extract(); 
    test_Extract_float(); 
    test_Trim(); 
    test_Count(); 
    test_All(); 
    test_Blank(); 
    test_ExtractLong(); 
    test_HeadFirst_HeadLast(); 
    test_FormatInt(); 
    test_LoadList(); 
    test_Format_Ellipsis(); 
    test_StartsWithLetterAZaz(); 
    test_FormatInt_2(); 
    test_ParseStringIntInt(); 
    */
    test_FormatIndex(); 


    return 0  ; 
}
// om-;TEST=SStrTest om-t
