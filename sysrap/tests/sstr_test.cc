// ~/opticks/sysrap/tests/sstr_test.sh

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include "sstr.h"

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




int main(int argc, char** argv)
{
    /*
    test_chop(); 
    test_StripTail(); 
    test_SideBySide(); 
    test_nullchar(true); 
    test_Write(); 
    test_empty(); 
    test_ParseIntSpecList<int64_t>() ; 
    test_ParseIntSpecList<int>() ; 
    test_ParseIntSpecList_demo<int>() ; 
    test_snprintf(); 
    */
    test_TAG(); 
   

    return 0 ; 
} 
// ~/opticks/sysrap/tests/sstr_test.sh


