#include "NState.hpp"
#include "Bookmarks.hh"
#include "InterpolatedView.hh"
#include <cstdio>
#include <cassert>


//  hmm need to collapse trackball state into View before saving state
//  for view based interpolation to operate

int main(int argc, char** argv)
{
    printf("%s::main \n", argv[0]);

    Bookmarks* bookmarks = new Bookmarks("$HOME/.opticks/juno/State");
    bookmarks->Summary();


    printf("%s::main aft bmks \n", argv[0]);


    //for(unsigned int i=0 ; i < 10 ; i++)
    //    bookmarks->number_key_pressed(i);

    InterpolatedView* iv = bookmarks->getInterpolatedView();
    assert(iv);

    iv->Summary();

    for(unsigned int i=0 ; i < 200 ; i++) iv->tick();


    return 0 ;
}

