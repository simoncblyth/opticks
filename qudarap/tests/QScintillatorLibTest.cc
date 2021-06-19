
#include "QTex2D.hh"

class GScintillatorLib ; 

struct QScintillatorLib
{
    const GScintillatorLib* lib ; 

    QScintillatorLib(const GScintillatorLib* lib_); 
    void init(); 
};


QScintillatorLib::QScintillatorLib(const GScintillatorLib* lib_)
    :
    lib(lib_)
{
    init(); 
}

void QScintillatorLib::init()
{
}


#include "Opticks.hh"
#include "GScintillatorLib.hh"
#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv); 
    ok.configure(); 

    GScintillatorLib* slib = GScintillatorLib::load(&ok);
    slib->dump();

    QScintillatorLib qlib(slib); 


    return 0 ; 
}

