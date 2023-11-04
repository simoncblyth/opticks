// ./sstr_test.sh

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


int main(int argc, char** argv)
{
    /*
    test_chop(); 
    test_StripTail(); 
    test_SideBySide(); 
    test_nullchar(true); 
    test_Write(); 
    test_empty(); 
    */


    return 0 ; 
} 
