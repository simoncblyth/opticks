// name=U4SensorIdentifierDefault_test ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <iostream>

bool IsInterestingCopyNo( int copyno )
{
    return 
        copyno > -1 && 
           (
            (std::abs( copyno -      0 ) < 100) || 
            (std::abs( copyno -  17612 ) < 100) ||
            (std::abs( copyno -  30000 ) < 100) ||
            (std::abs( copyno -  32400 ) < 100) ||
            (std::abs( copyno - 300000 ) < 100) || 
            (std::abs( copyno - 325600 ) < 100)  
           )
        ;   
}


int main()
{
    int count = 0 ; 
    for(int i=0 ; i < 400000 ; i++) 
    {
       bool select = IsInterestingCopyNo(i) ; 
       if(select) count += 1  ; 
       if(select) std::cout << i << std::endl ; 
    }
    std::cout << " count " << count << std::endl ; 
}

