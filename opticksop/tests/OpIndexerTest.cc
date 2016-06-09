#include "OpIndexerApp.hh"

int main(int argc, char** argv)
{
    OpIndexerApp app(argc, argv) ;

    app.configure();

    app.loadEvtFromFile();

    app.makeIndex();

    return 0 ; 
}
