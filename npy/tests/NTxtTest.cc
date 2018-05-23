#include "NTxt.hpp"

#include <cstdlib>
#include <cstring>

#include "BFile.hh"

#include "NPY_LOG.hh"
#include "BRAP_LOG.hh"
#include "PLOG.hh"

void test_read()
{
    char* idp = getenv("IDPATH") ;
    char path[256];
    snprintf(path, 256, "%s/GItemList/GMaterialLib.txt", idp );

    NTxt txt(path);
    txt.read();
}

void test_write()
{
    std::string x = BFile::FormPath("$TMP", "some/deep/reldir", "x.txt");
    std::string y = BFile::FormPath("$TMP", "some/deep/reldir", "y.txt");
    LOG(info) << "test_write " << x ; 

    NTxt tx(x.c_str());
    tx.addLine("one-x");
    tx.addLine("two");
    tx.addLine("three");
    tx.write();

    NTxt ty(y.c_str());
    ty.addLine("one-y");
    ty.addLine("two");
    ty.addLine("three");
    ty.write();

}


int main(int argc, char** argv )
{
    PLOG_(argc, argv);
    BRAP_LOG__ ;
    NPY_LOG__ ;

    test_write();


    return 0 ; 
}
