#include "sresource.h"
#include <iostream>

int main() {

    sresource::memlock::Limit mll = sresource::memlock::get();
    std::cout << mll.desc() << "\n" ;

    return 0 ;
}


