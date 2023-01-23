// name=sstr_test ; gcc $name.cc -g -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

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


int main(int argc, char** argv)
{
    /*
    test_chop(); 
    */

    test_StripTail(); 


    return 0 ; 
} 
