// name=st_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

#include <iostream>
#include <iomanip>
#include "st.h"

int main(int argc, char** argv)
{
    for(int h=0 ; h < 16 ; h++) 
    {
       int n = st::complete_binary_tree_nodes(h) ;  
       std::cout 
           << " h " << std::setw(3) << h 
           << " st::complete_binary_tree_nodes(h) " << std::setw(10) << n 
           << std::endl 
           ;
    }
    return 0 ; 
}
