#include "NState.hpp"
#include "Bookmarks.hh"
#include <cstdio>

int main(int argc, char** argv)
{
    NState* state = new NState("$HOME/.opticks/rainbow/State", "state" );
    Bookmarks* bookmarks = new Bookmarks(state) ; 

    for(unsigned int i=0 ; i < 10 ; i++)
    {
        bookmarks->number_key_pressed(i);
    }

    return 0 ;
}

