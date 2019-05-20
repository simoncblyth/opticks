#include "BTxt.hh"

#include <cstdlib>
#include <cstring>

#include "BFile.hh"

#include "OPTICKS_LOG.hh"

void test_read()
{
    char* idp = getenv("IDPATH") ;
    char path[256];
    snprintf(path, 256, "%s/GItemList/GMaterialLib.txt", idp );

    BTxt txt(path);
    txt.read();
}

void test_write()
{
    std::string x = BFile::FormPath("$TMP", "some/deep/reldir", "x.txt");
    std::string y = BFile::FormPath("$TMP", "some/deep/reldir", "y.txt");
    LOG(info) << "test_write " << x ; 

    BTxt tx(x.c_str());
    tx.addLine("one-x");
    tx.addLine("two");
    tx.addLine("three");
    tx.write();

    BTxt ty(y.c_str());
    ty.addLine("one-y");
    ty.addLine("two");
    ty.addLine("three");
    ty.write();

}


int main(int argc, char** argv )
{
    OPTICKS_LOG(argc, argv);

    test_write();


    return 0 ; 
}
