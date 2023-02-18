// name=s_pool_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include "Obj.h"
Obj::POOL Obj::pool = {} ; 

#include "_Obj.h"


int main()
{
    //std::vector<Obj> oo = {100,200,300} ;   // HUH: why does this assert but the below does not ?
    std::vector<_Obj> buf ; 
    {
        Obj a(100); 
        Obj b(200); 
        Obj c(300); 

        Obj::pool.serialize(buf) ; 
    }

    std::cout << " buf.size " << buf.size() << std::endl ; 

    Obj::pool.import(buf); 



    return 0 ; 
}
