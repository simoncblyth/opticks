#include "Composition.hh"
#include "Bookmarks.hh"
#include "View.hh"

#include <iostream>




int main(int argc, char** argv)
{
    if(argc < 2)
    {
        printf("%s : requires path to folder containing %s \n", argv[0], Bookmarks::filename );
        return 1 ; 
    }

    Composition composition ; 
    View* view = composition.getView();
    view->Summary("default view");

    Bookmarks bookmarks ; 
    bookmarks.setComposition(&composition);

    bookmarks.load(argv[1]);

    bookmarks.dump(1);
    bookmarks.apply(1);

    view->Summary("after apply(1)");


    view->setEye(0.,1.,2.);
    bookmarks.add(1);

    bookmarks.dump(1);


    return 0 ;
}

