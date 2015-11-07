#include "GMesh.hh"

int main(int argc, char** argv)
{
    GMesh* mesh = GMesh::load("/tmp/mm");

    mesh->Summary("check mesh loading");
    mesh->dump("mesh dump", 10);

    return 0 ;
}
