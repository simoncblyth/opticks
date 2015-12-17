#include "CfG4.hh"

int main(int argc, char** argv)
{
    CfG4* app = new CfG4("GGEOVIEW_") ; // TODO: change prefix to "OPTICKS_" to reflect generality 

    app->configure(argc, argv);
    app->propagate();
    app->save();

    delete app ; 
    return 0 ; 
}

