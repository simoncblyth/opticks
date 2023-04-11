#include <iostream>
#include <iomanip>

#include "U4RandomMonitor.h"


int main(int argc, char** argv)
{
    U4RandomMonitor mon ; 

    for(int i=0 ; i < 10 ; i++)
    {
        double u = G4UniformRand() ; 
        std::cout << std::setw(10) << std::setprecision(5) << u << std::endl ;
    }


    return 0 ; 
} 
/**

  0.13049
   0.61775
   0.99595
    0.4959
   0.11292
   0.28987
   0.47304
   0.83762
   0.35936
   0.92694


U4RandomMonitor::flat 0.13049
   0.13049
U4RandomMonitor::flat 0.617751
   0.61775
U4RandomMonitor::flat 0.995947
   0.99595
U4RandomMonitor::flat 0.495902
    0.4959
U4RandomMonitor::flat 0.112917
   0.11292
U4RandomMonitor::flat 0.289871
   0.28987
U4RandomMonitor::flat 0.473044
   0.47304
U4RandomMonitor::flat 0.837619
   0.83762
U4RandomMonitor::flat 0.359356
   0.35936
U4RandomMonitor::flat 0.926938
   0.92694


**/
