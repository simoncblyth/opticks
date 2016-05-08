#include "CCfG4.hh"

int main(int argc, char** argv)
{
    CCfG4* app = new CCfG4(argc, argv) ;

    app->interactive(argc, argv);

    app->propagate();
    app->save();

    delete app ; 
    return 0 ; 
}

