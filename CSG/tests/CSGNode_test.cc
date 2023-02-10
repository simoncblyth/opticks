// ./CSGNode_test.sh 

#include <iostream>
#include <vector>

#include "scuda.h"
#include "squad.h"
#include "CSGNode.h"

void test_Zero()
{
    CSGNode n0 = {} ; 
    std::cout << " CSGNode n0 = {} ; n0.desc()              : " << n0.desc() << std::endl ; 

    std::cout << " n0 " << n0 << std::endl ; 

    CSGNode nz = CSGNode::Zero(); 
    std::cout << " CSGNode nz = CSGNode::Zero() ; nz.desc() : " << nz.desc() << std::endl ; 

    std::cout << " nz " << nz << std::endl ; 

    std::vector<CSGNode> nn(15) ; 
    std::cout << " std::vector<CSGNode> nn(15) ; nn[0].desc() .... " << std::endl ; 
    for(int i=0 ; i < int(nn.size()) ; i++)  
    {
        const CSGNode& n = nn[i] ; 
        std::cout << " n.desc " << n.desc() << std::endl ; 
        std::cout << " n " << n << std::endl ; 
    }

    // subNum subOffset are defaulted to -1 for non-compound 
}


void test_Slot()
{
    std::vector<CSGNode> nn(15) ; 

    nn[8] = CSGNode::Sphere(100.f); 
    nn[14] = CSGNode::Box3(1000.f); 

    
    for(int i=0 ; i < int(nn.size()) ; i++)  
    {
        const CSGNode& n = nn[i] ; 
        std::cout << " i " << i << std::endl ;   
        std::cout << " n.desc " << n.desc() << std::endl ; 
        std::cout << " n " << n << std::endl ; 
    }
}


int main(int argc, char** argv)
{
    /*
    test_Zero(); 
    */

    test_Slot(); 


    return 0 ; 
}
