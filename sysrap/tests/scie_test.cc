#include "scie.h"

#include <iostream>
#include <iomanip>

int main()
{
    for( int w=380 ; w <= 750 ; w += 10 )
    {
        float wl = w ;
        float3 xyz = scie::xyzFit_1931(wl);
        std::cout
           << std::setw(4) << w
           << " ("
           << std::setw(10) << std::fixed << std::setprecision(3) << xyz.x
           << std::setw(10) << std::fixed << std::setprecision(3) << xyz.y
           << std::setw(10) << std::fixed << std::setprecision(3) << xyz.z
           << ")\n"
           ;
    }
}
