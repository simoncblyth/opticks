/**

~/o/sysrap/tests/SGLM_View_test.sh

**/


#include "SGLM_InterpolatedView.h"

int main()
{
     SGLM_View view = {} ;

     SGLM_InterpolatedView* iv = SGLM_InterpolatedView::Load("$FOLD/SGLM_View_test.npy") ;
     iv->setControlledView(&view);

     for(int i=0 ; i < 1000 ; i++)
     {
         iv->tick();
         std::cout << view.desc() << "\n" ;
     }
     return 0 ;
}
