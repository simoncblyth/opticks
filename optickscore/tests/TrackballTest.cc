#include "NGLM.hpp"
#include "Trackball.hh"

void TrackballTest()
{
   Trackball tb;
   tb.setRadius(0.8f);
   tb.Summary("init");

   tb.drag_to(0.f,0.f,0.01f,0.f);
   tb.Summary("after dx 0.01");
}


int main()
{
   TrackballTest();
   return 0 ;
}
