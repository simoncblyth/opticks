#include "Parameters.hpp"


int main()
{

    Parameters p ;
    p.add<int>("hello", 1);
    p.add<int>("world", 2);

    p.dump();

    return 0 ; 
}
