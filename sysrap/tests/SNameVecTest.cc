// om-;TEST=SNameVecTest om-t 

#include <cstring>
#include "SNameVec.hh"
#include "OPTICKS_LOG.hh"

struct Demo
{
    Demo(const char* name_) :  name(strdup(name_)) {} 
    const char* GetName() const ; 
    const char* name ; 
};

const char* Demo::GetName() const 
{
    return name ; 
}

template struct SNameVec<Demo> ; 

void test_Dump()
{
    std::vector<Demo*> a ; 
    a.push_back(new Demo("red")); 
    a.push_back(new Demo("green")); 
    a.push_back(new Demo("blue")); 
    a.push_back(new Demo("cyan")); 
    a.push_back(new Demo("magenta")); 
    a.push_back(new Demo("yellow")); 
    SNameVec<Demo>::Dump(a); 
}

void test_Sort()
{
    std::vector<Demo*> a ; 
    a.push_back(new Demo("red")); 
    a.push_back(new Demo("green")); 
    a.push_back(new Demo("blue")); 
    a.push_back(new Demo("cyan")); 
    a.push_back(new Demo("magenta")); 
    a.push_back(new Demo("yellow")); 

    LOG(info) << " Dump asis " ;  
    SNameVec<Demo>::Dump(a); 

    LOG(info) << " Sort " ;  
    SNameVec<Demo>::Sort(a, false);
    SNameVec<Demo>::Dump(a); 

    LOG(info) << " Sort reversed " ;  
    SNameVec<Demo>::Sort(a, true); 
    SNameVec<Demo>::Dump(a); 
}

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    //test_Dump(); 
    test_Sort(); 

    return 0 ;
}

// om-;TEST=SNameVecTest om-t 
