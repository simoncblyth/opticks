#include "GMergedMesh.hh"

int main(int argc, char** argv)
{
    GMergedMesh* mm = GMergedMesh::load("/tmp/mm");

    mm->Summary("mm loading");
    mm->Dump("mm dump", 10);
    mm->dumpSolids("dumpSolids");

    return 0 ;
}
