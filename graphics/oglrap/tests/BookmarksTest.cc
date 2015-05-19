#include "Composition.hh"
#include "Bookmarks.hh"
#include "View.hh"
#include "Scene.hh"

#include "stdio.h"


int main(int argc, char** argv)
{
    if(argc < 2)
    {
        printf("%s : requires path to folder containing %s \n", argv[0], Bookmarks::filename );
        return 1 ; 
    }

    Scene scene ; 
    Composition composition ; 

    View* view = composition.getView();
    view->Summary("default view");

    Bookmarks bookmarks ; 
    bookmarks.setComposition(&composition);
    bookmarks.setScene(&scene);

    bookmarks.load(argv[1]);

    //bookmarks.dump(1);
    //bookmarks.apply(1);


    for(unsigned int i=0 ; i < 10 ; i++)
    {
        bookmarks.number_key_pressed(i, 0);
    }

/*
    view->Summary("after apply(1)");

    view->setEye(0.,1.,2.);
    unsigned int changes = bookmarks.collect(1);
    printf("changes %u \n", changes );

    bookmarks.dump(1);
*/

    return 0 ;
}

