// ~/opticks/sysrap/tests/sstr_test.sh

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include "sstr.h"
#include "ssys.h"

std::vector<std::string> STRS = {{ "Hello__World", "Hello" , "Hello__" , "__World" }} ; 
const char* BLANK = "" ; 

void test_Chop()
{
    const char* str = "Hello__World" ; 
    std::pair<std::string, std::string> head_tail ; 
    sstr::Chop(head_tail, "__", str ); 

    std::cout << " head " << head_tail.first << std::endl ; 
    std::cout << " tail " << head_tail.second << std::endl ; 
}

void test_chop()
{
    for(unsigned i=0 ; i < STRS.size() ; i++ )
    {
        const char* str = STRS[i].c_str() ; 

        char* head ; 
        char* tail ; 
        sstr::chop(&head, &tail, "__", str ); 

        bool head_blank = head && strcmp( head, BLANK ) == 0 ; 
        bool tail_blank = tail && strcmp( tail, BLANK ) == 0 ; 

        std::cout 
            << " i " << std::setw(3) << i 
            << " str " << std::setw(30) << str 
            << " head [" << ( head ? head : "-" ) << "]" << " head_blank " << head_blank
            << " tail [" << ( tail ? tail : "-" ) << "]" << " tail_blank " << tail_blank
            << std::endl 
            ; 
    }
}


void test_HasTail_0()
{

    const char* lines = R"LIT(
0x
0xc0ffee
World0xc0ffee
World0xdeadbeef
World0xdead0xbeef
)LIT";
    std::stringstream ss(lines); 
    std::string l ;
    std::vector<std::string> v ; 
    while (std::getline(ss, l, '\n'))
    {
        if(l.empty()) continue ; 
        assert( sstr::HasTail(l) == true );     
        v.push_back(l);    
    }

    std::cout << "test_HasTail_0 v.size " << v.size() << "\n" ; 
    assert( sstr::HasTail(v) == true ); 
}



void test_HasTail_1()
{

    const char* lines = R"LIT(
red
green
blue
)LIT";
    std::stringstream ss(lines); 
    std::string l ;
    std::vector<std::string> v ; 
    while (std::getline(ss, l, '\n'))
    {
        if(l.empty()) continue ; 
        assert( sstr::HasTail(l) == false );     
        v.push_back(l);    
    }

    std::cout << "test_HasTail_1 v.size " << v.size() << "\n" ; 
    assert( sstr::HasTail(v) == false ); 
}



void test_HasTail()
{
    test_HasTail_0(); 
    test_HasTail_1(); 
}













void test_StripTail()
{

    const char* lines = R"LIT(
0x
0xc0ffee
World0xc0ffee
World0xdeadbeef
World0xdead0xbeef
)LIT";
    std::stringstream ss(lines); 
    std::string l ;
    while (std::getline(ss, l, '\n'))
    {   
        if(l.empty()) continue ; 
        std::string s = sstr::StripTail(l, "0x") ; 
        std::cout 
            << " l:" << std::setw(30) << l 
            << " s:" << std::setw(30) << s
            << " s.size: " << s.size()
            << " s.empty: " << s.empty()
            << std::endl
            ; 
    }
}


void test_StripComment()
{
    const char* lines = R"LIT(
hello:27:-1
hello:27:-1  ## some comment # 
)LIT";
    std::stringstream ss(lines); 
    std::string l ;
    while (std::getline(ss, l, '\n'))
    {   
        if(l.empty()) continue ; 
        std::string s = sstr::StripComment(l) ; 
        std::cout 
            << " l:[" << l << "]\n"
            << " s:[" << s << "] "
            << " s.size: " << s.size()
            << " s.empty: " << s.empty()
            << std::endl
            ; 
    }
}

void test_TrimString()
{
    const char* lines = R"LIT(
hello:27:-1
(   hello:27:-1    )
hello:27:-1  ## some comment # 
)LIT";
    std::stringstream ss(lines); 
    std::string l ;
    while (std::getline(ss, l, '\n'))
    {   
        if(l.empty()) continue ; 
        std::string s = sstr::TrimString(l) ; 
        std::cout 
            << " l:[" << l << "]\n"
            << " s:[" << s << "] "
            << " s.size: " << s.size()
            << " s.empty: " << s.empty()
            << std::endl
            ; 
    }
}






struct fspec
{
    static std::string Format(const std::vector<std::string>& fields ); 
    static void Init(std::vector<fspec>& fss, const std::vector<std::string>& fields ); 
    static fspec Init(const char* field); 

    std::vector<std::string> lines ; 
    int linecount ; 
    int maxlen ; 

    std::string desc() const ; 
};

inline std::string fspec::desc() const 
{
    std::stringstream ss ; 
    ss << " fspec::desc linecount " << linecount << " maxlen " << maxlen ;  
    std::string str = ss.str(); 
    return str ; 
}

inline fspec fspec::Init(const char* field)
{
    std::stringstream ss(field); 
    std::string l ;

    fspec fs = {} ; 

    fs.linecount = 0 ; 
    fs.maxlen = 0 ; 

    while (std::getline(ss, l, '\n'))
    {   
        fs.lines.push_back(l); 
        fs.linecount += 1 ;
        int len = l.size() ; 
        fs.maxlen = std::max( fs.maxlen, len );  
    }
    return fs ; 
}

inline void fspec::Init(std::vector<fspec>& fss, const std::vector<std::string>& fields )
{
    int num_fields = fields.size();   
    for(int i=0 ; i < num_fields ; i++)
    {
        const std::string& field = fields[i] ; 
        fspec fs = fspec::Init(field.c_str()); 
        fss.push_back(fs); 
    }
}

inline std::string fspec::Format(const std::vector<std::string>& fields )
{
    std::vector<fspec> fss ; 
    Init(fss, fields); 
   
    int num_fields = fields.size();   
    std::stringstream ss ; 
    ss << " num_fields " << num_fields << std::endl ; 
    for(int i=0 ; i < num_fields ; i++) ss << fss[i].desc() << std::endl ;  

    std::string str = ss.str(); 
    return str ; 
}






void test_SideBySide()
{
    const char* lines = R"LIT(
0x
0xc0ffee
World0xc0ffee
World0xdeadbeef
World0xdead0xbeef
)LIT";

    std::vector<std::string> fields ; 
    fields.push_back(lines); 
    fields.push_back(lines); 
    fields.push_back(lines); 

    std::cout << fspec::Format( fields ) << std::endl ; 


}


void test_nullchar(bool flip)
{
    char prefix = flip ? 'X' : '\0' ;
    std::string name ; 
    name += prefix ; 
    name += "hello" ; 

    std::cout << "[" << name << "]" << std::endl ; 
}

void test_Write()
{
    const char* path = "/tmp/test_Write.txt" ; 
    sstr::Write(path, "test_Write" ); 
}


void test_empty()
{
    std::string empty ; 
    std::cout << "empty         [" << empty << "]" << std::endl ; 
    std::cout << "empty.c_str() [" << empty.c_str() << "]" << std::endl ; 

    char c = empty.c_str()[0] ; 
    std::cout << "c [" << c << "]" << std::endl ; 

    bool c_is_terminator = c == '\0' ; 
    std::cout << " c_is_terminator " << ( c_is_terminator ? "YES" : "NO " ) << std::endl; 
}


template<typename T>
void test_ParseIntSpecList()
{
     const char* _spec = "1,2,3,100,200,h1,h5,6,7,K1,K10,11,12,M1,2,3,K1,2,M1,H1,2,h1:4" ; 

     std::vector<T> expect = {1,2,3,100,200,100,500,600,700,1000,10000,11000,12000,1000000,2000000,3000000,1000,2000,1000000,100000,200000,100,200,300,400 } ;  
     int num_expect = expect.size(); 

     std::vector<std::string> spec ; 
     sstr::Split( _spec, ',' , spec ); 
     int num_spec = spec.size(); 

     std::vector<T> value ; 
     sstr::ParseIntSpecList<T>(value, _spec); 
     int num_value = value.size(); 

     std::vector<T>* ls = sstr::ParseIntSpecList<T>(_spec) ;
     int num_ls = ls->size(); 

     std::cout 
         << " _spec " << std::endl
         << _spec 
         << std::endl 
         << " num_spec " << num_spec 
         << " num_value " << num_value
         << " num_expect " << num_expect
         << " num_ls " << num_ls
         << std::endl 
         ;

     assert( num_spec  <= num_expect ); 
     assert( num_value == num_expect ); 
     assert( num_ls == num_expect );

     int pass = 0 ; 
     for(int i=0 ; i < num_value ; i++)
     {
         const char* s = i < num_spec ? spec[i].c_str() : nullptr ; 
         T e = i < num_expect ? expect[i] : -1 ;  
         T v = i < num_value  ? value[i]  : -1 ;  
         T l = i < num_ls ?      (*ls)[i] : -1 ;  

         bool match = e == v && e == l ; 

         pass += int(match) ; 
         std::cout  
              << std::setw(10) << ( s ? s : "-" )
              << std::setw(10) << e
              << std::setw(10) << v
              << std::setw(10) << l
              << ( match ? " " : " ERROR MISMATCH" )
              << std::endl 
              ;

     }
     assert( pass == num_value ); 
}

/**

epsilon:opticks blyth$ ~/opticks/sysrap/tests/sstr_test.sh
        M1:5,K1:2 :  [1000000 2000000 3000000 4000000 5000000 1000 2000  ] 
  M1,2,3,4,5,K1,2 :  [1000000 2000000 3000000 4000000 5000000 1000 2000  ] 
            h1:10 :  [100 200 300 400 500 600 700 800 900 1000  ] 
            K1:10 :  [1000 2000 3000 4000 5000 6000 7000 8000 9000 10000  ] 
            H1:10 :  [100000 200000 300000 400000 500000 600000 700000 800000 900000 1000000  ] 
            M1:10 :  [1000000 2000000 3000000 4000000 5000000 6000000 7000000 8000000 9000000 10000000  ] 

**/


template<typename T>
void test_ParseIntSpecList_demo()
{
    std::vector<std::string> spec = { 
         "M1:5,K1:2" , 
         "M1,2,3,4,5,K1,2", 
         "h1:10",
         "K1:10",
         "H1:10",
         "M1:10"
     };  
    int num_spec = spec.size(); 
     
    std::vector<T> value ; 
    for(int i=0 ; i < num_spec ; i++)
    {
        const char* _spec = spec[i].c_str(); 
        sstr::ParseIntSpecList<T>(value, _spec);  
        std::cout 
            << std::setw(30) << _spec 
            << " : "
            << " [" 
            ; 
            
        int num_value = value.size(); 
        for(int i=0 ; i < num_value ; i++) std::cout << value[i] << " " ; 
        std::cout << " ] " << std::endl ;  
    }
}




void test_snprintf()
{
    char buf[10];
    for(int i=0 ; i < 1010 ; i++)
    {
        int n = snprintf(buf, 10, "%0.3d", i) ; 
        std::cout << buf << ":" << n << std::endl ; 
    }
} 



struct Prof
{
    static constexpr const char* FMT = "%0.3d" ; 
    static constexpr const int N = 10 ; 
    static char TAG[N] ; 
    static int SetTag(int idx, const char* fmt=FMT ); 
    static void UnsetTag(); 
};

char Prof::TAG[N] = {} ; 

inline int Prof::SetTag(int idx, const char* fmt)
{
    return snprintf(TAG, N, fmt, idx ); 
}
inline void Prof::UnsetTag()
{
    TAG[0] = '\0' ; 
}

void test_TAG()
{
    std::cout << __FUNCTION__ << std::endl; 
    std::cout << "[" << Prof::TAG << "]" << std::endl ; 
    for(int i=0 ; i < 100 ; i++) 
    {
        Prof::SetTag(i,"A%0.3d"); 
        if( i % 10 == 0 ) Prof::UnsetTag(); 
        std::cout << "[" << Prof::TAG << "]" << std::endl ; 
    }
}

void test_StripTail_Unique_0()
{
    std::vector<std::string> src = {{
        "red",
        "green",
        "blue",
        "cyan",
        "yellow",
        "magenta",
        "cyan",
        "cyan",
        "cyan",
    }}; 

    std::vector<std::string> key ; 
    sstr::StripTail_Unique( key, src ); 
    std::cout << sstr::DescKeySrc(key, src) ; 
}

void test_StripTail_Unique_1()
{
    std::vector<std::string> src = {{
        "red0xbeef",
        "green0xbeef",
        "blue0xbeef",
        "red0xbeef",
        "red0xbeef",
        "cyan",
        "yellow",
        "red0xbeef",
        "green0xbeef",
        "green0xbeef",
        "green0xbeef",
        "magenta",
        "green0xbeef",
        "red0xbeef",
        "red0xbeef",
        "red0xbeef",
        "red0xbeef",
        "red0xbeef",
        "red0xbeef",
    }} ; 

    std::vector<std::string> key ; 
    sstr::StripTail_Unique( key, src ); 
    std::cout << sstr::DescKeySrc(key, src) ; 
}

void test_Extract()
{
    std::vector<std::string> src = {{
           "red0xbeef",
           "BoxGridMultiUnion10:30_YX",
           "BoxGridMultiUnion10_30_YX"
    }} ; 

    for(unsigned i=0 ; i < src.size() ; i++)
    {
        const char* st = src[i].c_str();   
        std::vector<long> vals ; 
        sstr::Extract(vals, st);  
        std::cout 
            << std::setw(20) << st 
            << " : "
            << vals.size()
            << "\n"
            ;
    }

}

void test_Concat()
{
    const char* s0 = sstr::Concat("aa","bb","cc","dd" ); 
    assert( strcmp(s0, "aabbccdd" ) == 0 ); 
    const char* s1 = sstr::Concat("aa","bb",nullptr,"dd" ); 
    assert( strcmp(s1, "aabbdd" ) == 0 ); 
}


struct sstr_test 
{
    static int Main(); 
};  


int sstr_test::Main()
{
    //const char* test = "TrimString" ; 
    const char* test = "StripComment" ; 

    const char* TEST = ssys::getenvvar("TEST", test ); 

    if(     strcmp(TEST, "HasTail")==0 )    test_HasTail(); 
    else if(strcmp(TEST, "chop")==0 )       test_chop(); 
    else if(strcmp(TEST, "StripTail")==0 )  test_StripTail(); 
    else if(strcmp(TEST, "StripComment")==0 )  test_StripComment(); 
    else if(strcmp(TEST, "TrimString")==0 )  test_TrimString(); 
    else if(strcmp(TEST, "SideBySide")==0 ) test_SideBySide(); 
    else if(strcmp(TEST, "nullchar")==0 )   test_nullchar(true); 
    else if(strcmp(TEST, "Write")==0 )      test_Write(); 
    else if(strcmp(TEST, "ParseIntSpecList64")==0 )   test_ParseIntSpecList<int64_t>();
    else if(strcmp(TEST, "ParseIntSpecList32")==0 )   test_ParseIntSpecList<int>();
    else if(strcmp(TEST, "ParseIntSpecListDemo")==0 ) test_ParseIntSpecList_demo<int>();
    else if(strcmp(TEST, "snprintf")==0 )             test_snprintf(); 
    else if(strcmp(TEST, "TAG")==0 )                  test_TAG(); 
    else if(strcmp(TEST, "StripTailUnique")==0 )      test_StripTail_Unique_0(); 
    else if(strcmp(TEST, "Extract") == 0 )            test_Extract(); 
    else if(strcmp(TEST, "Concat") == 0 )             test_Concat(); 

    return 0 ; 
}

int main(int argc, char** argv)
{
    return sstr_test::Main() ; 
} 
// ~/opticks/sysrap/tests/sstr_test.sh


