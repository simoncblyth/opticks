#include "srpn_tranche.h"

int main()
{
    srpn_tranche tr ;
    std::cout << tr.desc() ;

    tr.push( 100, 200, 1.5f );
    std::cout << tr.desc() ;

    tr.push( 200, 300, 2.5f );
    std::cout << tr.desc() ;




    return 0 ;
}
