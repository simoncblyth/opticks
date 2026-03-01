/**

~/o/sysrap/tests/SGLM_View_test.sh

**/


#include "SGLM_InterpolatedView.h"

int main()
{
     SGLM_View view = {} ;
     std::cout << view.desc() << "\n" ;
     assert( view.is_zero() );


     SGLM_InterpolatedView* interpolated_view = SGLM_InterpolatedView::Load("$FOLD/SGLM_View_test.npy") ;
     interpolated_view->setControlledView(&view);

     for(int i=0 ; i < 1000 ; i++)
     {
         interpolated_view->tick();
         std::cout << view.desc() << "\n" ;
         assert( !view.is_zero() );
     }

     view.set_zero();
     std::cout << view.desc() << "\n" ;

     return 0 ;
}
