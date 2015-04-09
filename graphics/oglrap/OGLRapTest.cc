#include "Trackball.hh"


int main()
{
   Trackball tb;
   tb.setRadius(0.8);
   tb.Summary("init");

   tb.drag_to(0,0,0.01,0);
   tb.Summary("after dx 0.01");

   return 0 ;
}
