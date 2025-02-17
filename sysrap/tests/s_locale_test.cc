/**

~/opticks/sysrap/tests/s_locale_test.sh

**/

#include <iostream>
#include <iomanip>
#include <locale>
#include <string>
#include <sstream>

int main()
{
    int v = 1000000000 ; 

    std::cout.imbue(std::locale(""));
    std::cout << std::setw(10) << v << std::endl;

    std::stringstream ss ;
    ss.imbue(std::locale("")) ;  // commas for thousands
    ss << std::setw(10) << v << std::endl; 

    std::string str = ss.str(); 
    std::cout << str ;  


    return 0 ; 
}

