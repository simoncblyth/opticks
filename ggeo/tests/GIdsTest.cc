#include "GIds.hh"

int main()
{
    GIds* t = new GIds() ; 
    for(unsigned int i=0 ; i < 10 ; i++) t->add(i,i+1,i+2,i+3) ;
        
    t->save("/tmp/ids.npy");

    return 0 ; 
}
