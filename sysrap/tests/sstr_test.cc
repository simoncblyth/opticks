/**
sstr_test.cc
==============

~/opticks/sysrap/tests/sstr_test.sh

TEST=Format ~/opticks/sysrap/tests/sstr_test.sh
TEST=ParseIntSpecList32   ~/opticks/sysrap/tests/sstr_test.sh
TEST=ParseIntSpecList64   ~/opticks/sysrap/tests/sstr_test.sh
TEST=ParseIntSpecListDemo ~/opticks/sysrap/tests/sstr_test.sh

**/

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include "sstr.h"
#include "ssys.h"





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
     const char* _spec = "1,2,3,100,200,h1,h5,6,7,K1,K10,11,12,M1,2,3,K1,2,M1,H1,2,G2,T1,2,h1:4" ;

     std::vector<T> expect = {                1,
                                              2,
                                              3,
                                            100,
                                            200,
                                            100,
                                            500,
                                            600,
                                            700,
                                          1'000,
                                         10'000,
                                         11'000,
                                         12'000,
                                      1'000'000,
                                      2'000'000,
                                      3'000'000,
                                          1'000,
                                          2'000,
                                      1'000'000,
                                        100'000,
                                        200'000,
                                  2'000'000'000,
                              1'000'000'000'000,
                              2'000'000'000'000,
                                            100,
                                            200,
                                            300,
                                            400
                                                };
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

     std::cout
          << std::setw(16) << "spec_input"
          << std::setw(16) << "expected"
          << std::setw(16) << "parsed"
          << std::setw(16) << "ls_parsed"
          << std::setw(20) << "match"
          << std::endl
          ;


     for(int i=0 ; i < num_value ; i++)
     {
         const char* s = i < num_spec ? spec[i].c_str() : nullptr ;
         T e = i < num_expect ? expect[i] : -1 ;
         T v = i < num_value  ? value[i]  : -1 ;
         T l = i < num_ls ?      (*ls)[i] : -1 ;

         bool match = e == v && e == l ;

         pass += int(match) ;
         std::cout
              << std::setw(16) << ( s ? s : "-" )
              << std::setw(16) << e
              << std::setw(16) << v
              << std::setw(16) << l
              << std::setw(20) << ( match ? " YES " : " NO  " )
              << std::endl
              ;

     }
     assert( pass == num_value );
}


template<typename T>
void test_ParseIntSpecList_demo()
{
    std::vector<std::string> spec = {
         "M1:5,K1:2" ,
         "M1,2,3,4,5,K1,2",
         "h1:10",
         "K1:10",
         "H1:10",
         "M1:10",
         "M1x10",
         "1x5,2x5",
         "1001"
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


void test_Format()
{
    const char* str_0 = sstr::Format("u_%d.npy", 214) ;
    const char* str_1 = sstr::Format("u_%llu.npy", 214ull ) ;

    std::cout
       << "test_Format\n"
       << "str_0:[" << str_0 << "]\n"
       << "str_1:[" << str_1 << "]\n"
       ;
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

void test_IsInteger()
{
    std::vector<std::string> src = {{
        "0",
        "1",
        "9",
        "10",
        "100",
        "1000",
        "",
        "-1",
        " 1"
    }} ;

    for(unsigned i=0 ; i < src.size() ; i++)
    {
        const char* st = src[i].c_str();
        bool ii = sstr::IsInteger(st) ;
        std::cout
            << ( ii ? "YES" : "NO " )
            << " : "
            << "[" << st << "]"
            << "\n"
            ;
    }
}



struct sstr_test
{
    static constexpr const uint64_t M = 1000000 ;
    static constexpr const uint64_t G = 1000000000 ;

    static constexpr const char* BLANK = "" ;
    static std::vector<std::string> STRS ;
    static std::vector<std::string> STRS2 ;

    static int Chop();
    static int chop();
    static int prefix_suffix();

    static int StartsWithElem();
    static int split();
    static int ParseInt();


    static int Main();
};


std::vector<std::string> sstr_test::STRS = {{ "Hello__World", "Hello" , "Hello__" , "__World" }} ;
std::vector<std::string> sstr_test::STRS2 = {{ "/tmp/w54.npy[0:1]", "/tmp/w54.npy[0:2]", "[0:1]" }} ;


int sstr_test::Chop()
{
    const char* str = "Hello__World" ;
    std::pair<std::string, std::string> head_tail ;
    sstr::Chop(head_tail, "__", str );

    std::cout << " head " << head_tail.first << std::endl ;
    std::cout << " tail " << head_tail.second << std::endl ;

    return 0;
}

int sstr_test::chop()
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
    return 0;
}



int sstr_test::prefix_suffix()
{

    for(unsigned i=0 ; i < STRS2.size() ; i++ )
    {
        const char* str = STRS2[i].c_str() ;
        char* pfx = nullptr ;
        char* sfx = nullptr ;
        bool has_suffix = sstr::prefix_suffix(&pfx, &sfx, "[", str );
        std::cout
            << "sstr_test::prefix_suffix\n"
            << " str [" << ( str ? str : "-" ) << "]\n"
            << " pfx [" << ( pfx ? pfx : "-" ) << "]\n"
            << " sfx [" << ( sfx ? sfx : "-" ) << "]\n"
            << " has_suffix " << ( has_suffix ? "YES" : "NO " ) << "\n"
            ;
    }
    return 0 ;
}


int sstr_test::StartsWithElem()
{
    std::cout << "[StartsWithElem\n" ;
    const char* s = "TO BT BT BT BT BR BT BT BT BT BT BT SC BT BT BT BT SD" ;

    bool starts_with_TO_CK_SI = sstr::StartsWithElem(s, "TO,CK,SI" );
    bool starts_with_XX_YY_ZZ = sstr::StartsWithElem(s, "XX,YY,ZZ" );

    assert( starts_with_TO_CK_SI == true );
    assert( starts_with_XX_YY_ZZ == false );
    std::cout << "]StartsWithElem\n" ;

    return 0;
}
int sstr_test::split()
{
     const char* spec = "1.1,0,0,0,0,1.2,0,0,0,0,1.3,0,0,0,0,1.4" ;
     std::vector<double> elem ;
     sstr::split<double>( elem, spec, ',' );

     std::cout << sstr::desc<double>(elem) ;

     return 0;
}


int sstr_test::ParseInt()
{
    std::vector<std::string> src = {{
        "M1",
        "G1",
        "G2",
        "G3",
        "X0",
        "X1",
        "X2",
        "X4",
        "X8",
        "X16",
        "X31",
        "X32",
        "X63",
        "X64"
    }} ;

    for(unsigned i=0 ; i < src.size() ; i++)
    {
        const char* st = src[i].c_str();
        uint64_t value = sstr::ParseInt<uint64_t>(st) ;
        std::cout
            << " spec  " << std::setw(5) << st
            << " value " << std::setw(20) << value
            << " value/M " << std::setw(20) << value/M
            << " value/G " << std::setw(20) << value/G
            << "\n"
            ;
    }
    return 0 ;
}



int sstr_test::Main()
{
    //const char* test = "TrimString" ;
    //const char* test = "StripComment" ;
    //const char* test = "IsInteger" ;
    //const char* test = "StartsWithElem" ;
    const char* test = "split" ;

    const char* TEST = ssys::getenvvar("TEST", test );
    bool ALL = strcmp(TEST, "ALL") == 0 ;


    if(     strcmp(TEST, "HasTail")==0 )    test_HasTail();
    else if(strcmp(TEST, "StripTail")==0 )  test_StripTail();
    else if(strcmp(TEST, "StripComment")==0 )  test_StripComment();
    else if(strcmp(TEST, "TrimString")==0 )  test_TrimString();
    else if(strcmp(TEST, "SideBySide")==0 ) test_SideBySide();
    else if(strcmp(TEST, "nullchar")==0 )   test_nullchar(true);
    else if(strcmp(TEST, "Write")==0 )      test_Write();
    else if(strcmp(TEST, "ParseIntSpecList64")==0 )   test_ParseIntSpecList<int64_t>();
    //else if(strcmp(TEST, "ParseIntSpecList32")==0 )   test_ParseIntSpecList<int>();
    else if(strcmp(TEST, "ParseIntSpecListDemo")==0 ) test_ParseIntSpecList_demo<int64_t>();
    else if(strcmp(TEST, "snprintf")==0 )             test_snprintf();
    else if(strcmp(TEST, "Format") == 0 )             test_Format();
    else if(strcmp(TEST, "TAG")==0 )                  test_TAG();
    else if(strcmp(TEST, "StripTailUnique")==0 )      test_StripTail_Unique_0();
    else if(strcmp(TEST, "Extract") == 0 )            test_Extract();
    else if(strcmp(TEST, "Concat") == 0 )             test_Concat();
    else if(strcmp(TEST, "IsInteger") == 0 )          test_IsInteger();


    int rc = 0 ;
    if(ALL||0==strcmp(TEST,"Chop"))           rc += Chop();
    if(ALL||0==strcmp(TEST,"chop"))           rc += chop();
    if(ALL||0==strcmp(TEST,"prefix_suffix"))  rc += prefix_suffix();
    if(ALL||0==strcmp(TEST,"StartsWithElem")) rc += StartsWithElem();
    if(ALL||0==strcmp(TEST,"split"))          rc += split();
    if(ALL||0==strcmp(TEST,"ParseInt"))       rc += ParseInt();

    return rc ;
}

int main(int argc, char** argv)
{
    return sstr_test::Main() ;
}
// ~/opticks/sysrap/tests/sstr_test.sh


