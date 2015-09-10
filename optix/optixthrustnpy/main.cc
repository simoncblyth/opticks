#include "App.hh"

int main()
{
    App app ; 
    app.loadGenstep();
    app.initOptiX();
    app.uploadGenstep();
    app.checkGenstep();

    return 0 ;
}
