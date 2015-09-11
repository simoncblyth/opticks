#include "App.hh"

int main()
{
    App app ; 
    app.loadGenstep();
    app.initOptiX();
    app.uploadEvt();
    app.dumpGensteps();
    app.dumpPhotons();

    return 0 ;
}
