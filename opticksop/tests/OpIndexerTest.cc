#include "OpIndexerApp.hh"

int main(int argc, char** argv)
{
    OpIndexerApp app ;

    app.configure(argc, argv);

    app.loadEvtFromFile();

    app.makeIndex();

    return 0 ; 
}
