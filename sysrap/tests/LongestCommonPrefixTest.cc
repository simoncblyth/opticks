// name=LongestCommonPrefixTest ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <algorithm>
#include <iostream>
#include <vector>
#include <string>

/**
translation from the python os.path.commonprefix 
**/

std::string commonprefix(const std::vector<std::string>& a)
{
    std::vector<std::string> aa(a); 
    std::sort( aa.begin(), aa.end() ); 
    const std::string& s1 = aa[0] ; 
    const std::string& s2 = aa[aa.size()-1] ; 
    for(unsigned i=0 ; i < s1.size() ; i++) if( s1[i] != s2[i] ) return s1.substr(0,i) ; 
    return s1 ; 
} 

void dump(const std::vector<std::string>& a)
{
    std::cout << "-----" << std::endl ; 
    for(unsigned i=0 ; i < a.size() ; i++ ) std::cout << a[i] << std::endl; 
}


int main()
{
    std::vector<std::string> a = { "one/z", "one/a", "one/b", "one/c" } ; 
    dump(a); 

    std::string cpfx = commonprefix(a); 
    std::cout << " cpfx " << cpfx << std::endl ; 

    dump(a); 


    return 0 ; 
}
