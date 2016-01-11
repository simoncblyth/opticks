#include "NState.hpp"
#include "Bookmarks.hh"
#include "InterpolatedView.hh"
#include <cstdio>

//  hmm need to collapse trackball state into View before saving state
//  for view based interpolation to operate

int main(int argc, char** argv)
{
    Bookmarks* bookmarks = new Bookmarks("$HOME/.opticks/juno/State");
    bookmarks->Summary();


    //for(unsigned int i=0 ; i < 10 ; i++)
    //    bookmarks->number_key_pressed(i);

    InterpolatedView* iv = bookmarks->getInterpolatedView();
    iv->Summary();

    for(unsigned int i=0 ; i < 200 ; i++) iv->tick();
    


    return 0 ;
}

