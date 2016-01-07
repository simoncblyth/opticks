#include "CfG4.hh"

int main(int argc, char** argv)
{
    CfG4* app = new CfG4(argc, argv) ;

    app->propagate();
    app->save();

    delete app ; 
    return 0 ; 
}

