#include "stree.h"

const char* BASE = getenv("BASE");  
int main(int argc, char** argv)
{
    stree st ; 
    st.level = 2 ; 
    st.load(BASE); 

    std::cout << st.desc() ; 

    return 0 ; 
}
