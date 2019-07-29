// TEST=BTxtTest om-t

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

void test_load()
{
    BTxt* txt = BTxt::Load("$TMP/ox_1872.log"); 
    LOG(info) << txt->desc();  

    unsigned ni = txt->getNumLines(); 

    for(unsigned i=0 ; i < ni ; i++)
    {
        const std::string& line = txt->getString(i); 

        std::size_t pu = line.find("u_") ; 
        if( pu == std::string::npos ) continue ;
        pu += 2 ;     

        std::size_t pc = line.find(":", pu) ; 
        if( pc == std::string::npos ) continue ;   
        pc += 1 ;  

        std::size_t ps = line.find(" ", pc) ; 
        if( ps == std::string::npos ) continue ;   

        std::string k = line.substr(pu,pc-pu-1); 
        std::string v = line.substr(pc,ps-pc); 

        LOG(info) << line ; 
        LOG(info) << "[" << k << "]" ;  
        LOG(info) << "[" << v << "]" ;  
    }
}




int main(int argc, char** argv )
{
    OPTICKS_LOG(argc, argv);

    //test_write();
    test_load(); 

    return 0 ; 
}
