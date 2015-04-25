#include "Bookmarks.hh"

#include <iostream>

int main(int argc, char** argv)
{
    Bookmarks bookmarks ; 
    if(argc > 1)
    {
        bookmarks.load(argv[1]);
    }
    else
    {
        std::cout << "Requires path to bookmarks in ini format" << std::endl ; 
    } 
    return 0 ;
}

