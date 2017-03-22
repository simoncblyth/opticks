#include "NOctNode.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ;  

   
    //int size = 1 << 5 ;   // 32
    int size = 1 << 4 ;   // 16

    nivec3 min(-size/2,-size/2,-size/2) ;

    float scale = 2.0f ;  // indice ints to real world floats 

    NOctNode* tree = NOctNode::Construct( min, size, scale); 

    NOctNode::Traverse(tree, 0);

    int num = NOctNode::TraverseIt(tree);

    LOG(info) << "NOctNode count " << num << " size*size*size " << size*size*size ;


    return 0 ; 
}


// hmm without simplification the Octress gains little
//  NOctNode count 1065 size*size*size 4096
//
