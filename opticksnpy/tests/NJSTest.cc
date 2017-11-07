
#include <string>
#include <map>

#include "NJS.hpp"
#include "NPY_LOG.hh"
#include "PLOG.hh"


using json = nlohmann::json;


void test_write_read()
{
    const char* path = "$TMP/NJSTest/test_write_read.json" ;

    std::map<std::string, int> m { {"one", 1}, {"two", 2}, {"three", 3} };
    json js(m) ; 
    NJS njs(js) ; 
    njs.write(path);

    NJS njs2 ; 
    njs2.read(path);
    njs2.dump();

    json& js2 = njs2.get() ;
    LOG(info) << "js2:" << js2.dump(4) ; 
}


int main(int argc, char** argv)
{

    PLOG_(argc, argv);
    NPY_LOG__ ; 


    test_write_read();



    return 0 ; 
}

