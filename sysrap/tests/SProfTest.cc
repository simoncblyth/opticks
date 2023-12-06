// name=SProfTest ; gcc $name.cc ../SProf.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <iostream> 
#include "SProf.hh"

struct SProfTest
{
    static void Add_Write_Read(); 
    static void Read(); 
    static void SetTag(); 
    static void Main(); 
}; 

inline void SProfTest::Add_Write_Read()
{
    std::cout << __FUNCTION__ << std::endl ; 

    SProf::Add("start"); 
    SProf::Add("red"); 
    SProf::SetTag(0, "A%0.3d_" ) ; 
    SProf::Add("green"); 
    SProf::SetTag(1, "A%0.3d_" ) ; 
    SProf::Add("blue"); 
    SProf::UnsetTag(); 
    SProf::Add("stop"); 

    std::cout << SProf::Desc() ; 
    const char* path = "/tmp/SProf.txt" ; 
    bool append = false ; 

    std::cout << "--------------------------------" << std::endl ; 

    SProf::Write(path, append ); 
    SProf::Read(path);  
    std::cout << SProf::Desc() ; 
}


inline void SProfTest::Read()
{
    const char* path = "run_meta.txt" ; 
    SProf::Read(path);  
    std::cout << SProf::Desc() ; 
}

inline void SProfTest::SetTag()
{
    for(int i=0 ; i < 100 ; i++)
    {
        SProf::SetTag(i, "A%0.3d_" ); 
        if(i % 10 == 0 ) SProf::UnsetTag(); 
        std::cout << "[" << SProf::TAG << "]" << std::endl ; 
    }
}
 


inline void SProfTest::Main()
{
    /*
    Read(); 
    SetTag(); 
    */
    Add_Write_Read() ;
}

int main()
{
    SProfTest::Main(); 
    return 0 ; 
}
