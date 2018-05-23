#include "NTreeTools.hpp"

#include "NPY_LOG.hh"
#include "PLOG.hh"


struct Node 
{  
   Node(int idx) : idx(idx) 
   {
       init();
   }
   void init(){ for(int i=0 ; i < 4 ; i++) children[i] = NULL ; }

   int idx ; 
   Node* children[4] ; 
};


template class NTraverser<Node,4> ; 


int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    NPY_LOG__ ; 

    Node root(0);
    root.children[0] = new Node(1) ; 
    root.children[3] = new Node(3) ; 

    NTraverser<Node,4> tv(&root, "NTraverser", 1 ) ; 


    return 0 ; 
}
