
// name=QBufTest ; gcc $name.cc -I.. -I/usr/local/cuda/include -L/usr/local/cuda/lib -I$HOME/np -std=c++11 -lstdc++ -lcudart -o /tmp/$name && /tmp/$name


#include <cassert>
#include <cuda_runtime.h>
#include "QBuf.hh"

int main(int argc, char** argv )
{
    unsigned num_items = 100 ; 

    QBuf<float>* buf = new QBuf<float>() ; 
    buf->device_alloc(num_items); 
    buf->device_set(0); 

    assert( buf); 
  
    return 0 ; 
}
