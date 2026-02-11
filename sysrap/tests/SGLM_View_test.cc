#include "SGLM_InterpolatedView.h"

int main()
{
     SGLM_InterpolatedView* iv = SGLM_InterpolatedView::Load("$FOLD/SGLM_View_test.npy") ;
     for(int i=0 ; i < 1000 ; i++)
     {
         iv->tick();
         std::cout << iv->desc() << "\n" ;
     }
     return 0 ;
}
