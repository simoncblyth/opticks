#include <vector>
#include <map>
#include <string>

#include "OPTICKS_LOG.hh"


struct Prop
{
    Prop(const char* name_, int value_)
        :
        name( strdup(name_) ),
        value(value_) 
    {
    }  
    const char* name ; 
    int value ; 
};



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    std::map<std::string, Prop*> pm ; 

    Prop* a = new Prop("yo1", 42 ) ; 
    Prop* b = new Prop("yo2", 43 ) ; 
    Prop* c = new Prop("yo3", 44 ) ; 

    pm["ri"] = a ; 
    pm["vg"] = b ; 
    pm["sc"] = c ; 


    std::string a_key = "ri" ; 
    std::string b_key = "vg" ; 
    std::string c_key = "sc" ; 

    assert( pm.at(a_key) == a ) ;
    assert( pm.at(b_key) == b ) ;
    assert( pm.at(c_key) == c ) ;
 
    return 0 ; 
}
