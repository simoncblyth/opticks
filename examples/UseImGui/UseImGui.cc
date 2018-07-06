#include <cassert>
#include <ImGui/imgui.h>




int main()
{
    float vx = 1.f ; 
    float vy = 2.f ; 

    ImVec2 v(vx,vy); 
    assert( v.x == vx ); 
    assert( v.y == vy ); 


    assert( ImGuiWindowFlags_NoTitleBar == 1 << 0 ); 

    return 0 ; 
}
