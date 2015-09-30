#include "DedupeNPY.hpp"
#include "NCache.hpp"

#include "stdlib.h"
#include "assert.h"

int main(int argc, char** argv)
{
    char* dir = getenv("JDPATH") ;
    NCache cache(dir);


    NPY<float>* vertices_ = NPY<float>::load( cache.path("GMergedMesh/0/vertices.npy").c_str() );  
    NPY<int>* faces_      = NPY<int>::load( cache.path("GMergedMesh/0/indices.npy").c_str() );  
    NPY<int>* nodes_      = NPY<int>::load( cache.path("GMergedMesh/0/nodes.npy").c_str() );  

    Ary<float> vertices( vertices_->getValues(), vertices_->getShape(0), 3 );
    Ary<int>   faces(       faces_->getValues(), faces_->getShape(0)/3, 3 ); // at NPY level indices have shape (3n, 1) rather than (n,3)





    return 0 ; 
}
 
