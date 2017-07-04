
#include "SSortKV.hh"
#include "PLOG.hh"

void SSortKV::dump(const char* msg) const 
{
    unsigned num = getNum();
    LOG(info) << msg << " num " << num ; 
    for(unsigned i=0 ; i < num ; i++)
    {
        std::string k = getKey(i);
        float v = getVal(i);

        std::cout 
            << " i " << std::setw(5) << i 
            << " v " << std::setw(10) << std::fixed << std::setprecision(3) << v 
            << " k " << std::setw(30) << k
            << std::endl
            ;

    }
}


