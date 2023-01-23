// name=SNameOrder_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <string>

struct Demo
{
    std::string name ; 
    Demo(const char* name); 
    const std::string& GetName() const ; 
}; 

inline Demo::Demo(const char* name_)
    :
    name(name_)
{
}
inline const std::string& Demo::GetName() const 
{
    return name ; 
}

#include "SNameOrder.h"



int main(int argc, char** argv)
{
    std::vector<Demo*> dd ; 
    dd.push_back( new Demo("Red0xc0ffee") ) ;
    dd.push_back( new Demo("Green0xc0ffee") );
    dd.push_back( new Demo("Blue0xc0ffee") ) ;
 
    std::cout << SNameOrder<Demo>::Desc(dd) << std::endl ;  

    SNameOrder<Demo>::Sort(dd) ; 

    std::cout << SNameOrder<Demo>::Desc(dd) << std::endl ;  


    return 0 ; 

}
