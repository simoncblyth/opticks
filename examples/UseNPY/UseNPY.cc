#include "NPY.hpp"

int main(int argc, char** argv)
{
    NPY<float>* a = NPY<float>::make_identity_transforms(10) ; 
    a->save("$TMP", "UseNPY.npy") ; 
    std::cout << "writing $TMP/UseNPY.npy" << std::endl ; 

    // python -c "import numpy as np ; print np.load(\"/tmp/$USER/opticks/UseNPY.npy\") "

    return 0 ; 
}
