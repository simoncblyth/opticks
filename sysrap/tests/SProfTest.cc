// name=SProfTest ; gcc $name.cc ../SProf.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <iostream> 
#include "SProf.hh"

struct SProfTest
{
    static void Add_Write_Read(); 
    static void Read(); 
    static void Main(); 
}; 

inline void SProfTest::Add_Write_Read()
{
    SProf::Add("start"); 
    SProf::Add("red"); 
    SProf::Add("green"); 
    SProf::Add("blue"); 
    SProf::Add("stop"); 

    std::cout << SProf::Desc() ; 
    const char* path = "/tmp/SProf.txt" ; 
    bool append = false ; 

    SProf::Write(path, append ); 
    SProf::Read(path);  
    std::cout << SProf::Desc() ; 
}

inline void SProfTest::Main()
{
    /*
    Add_Write_Read() ;
    */
    Read(); 
}

inline void SProfTest::Read()
{
    const char* path = "run_meta.txt" ; 
    SProf::Read(path);  
    std::cout << SProf::Desc() ; 
}


int main()
{
    SProfTest::Main(); 
    return 0 ; 
}
