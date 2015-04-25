#include "Geometry.hh"
#include "GMergedMesh.hh"

int main(int argc, char** argv)
{
    Geometry geometry ; 
    geometry.load("GGEOVIEW_");

    GMergedMesh* mm = geometry.getMergedMesh();
    mm->dumpSolids(); 


    return 0 ;
}

