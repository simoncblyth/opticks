/**
SGLM_SmoothNormals_test.cc
=============================

::

    ~/o/sysrap/tests/SGLM_SmoothNormals_test.sh

**/

#include "NPFold.h"
#include "SGLM.h"

int main()
{
    NPFold* fold = NPFold::Load("$MESH_FOLD"); 

    const NP* a_vtx = fold->get("vtx"); 
    const NP* a_tri = fold->get("tri"); 

    NP* a_nrm = SGLM::SmoothNormals( a_vtx, a_tri ); 

    a_nrm->save("$MESH_FOLD/nrm.npy") ; 

    return 0 ;  
}
