#include "stree.h"

const char* BASE = getenv("BASE");  
int main(int argc, char** argv)
{
    stree st ; 
    std::cout << " st.level " << st.level << std::endl ; 

    st.load(BASE); 
    std::cout << st.desc() ; 



    return 0 ; 
}
