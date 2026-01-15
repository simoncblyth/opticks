#include <iostream>
#include "sregex.h"

int main()
{
    std::vector<std::string> names = {
        "s_EMFcoil_holder_ring25_seg20",
        "s_EMFcoil_holder_ring2_seg0",
        "invalid_name",
        "s_EMFcoil_holder_ring26_seg10",
        "s_EMFcoil_holder_ring26_seg11",
        "s_EMFcoil_holder_ring26_seg12",
        "s_EMFcoil_holder_ring26_seg13",
        "s_EMFcoil_holder_ring26_seg14",
        "s_EMFcoil_holder_ring26_seg15",
        "s_EMFcoil_holder_ring_mod26_seg1",
        "s_EMFcoil_holder_ring_mod26_seg2",
        "sWorld",
        "s_EMFcoil_not_quite",
        "s_EMFcoil_holder_ring27_seg1",
        "s_EMFcoil_holder_ring27_seg2",
        "s_EMFcoil_holder_ring27_seg3",
        "s_EMFcoil_holder_ring27_seg4",
        "s_EMFcoil_holder_ring_mod27_seg1",
        "s_EMFcoil_holder_ring_mod27_seg2",
        "s_EMFcoil_holder_ring28_seg1",
        "s_EMFcoil_holder_ring28_seg2",
        "s_EMFcoil_holder_ring28_seg3",
        "s_EMFcoil_holder_ring28_seg4",
        "s_EMFcoil_holder_ring28_seg5",
        "s_EMFcoil_holder_ring28_seg6"
    };

    sregex ptn("REGEX");
    for (const auto& name : names)
        std::cout << ( ptn.matches(name) ? "YES" : "NO " )  << " : " << name << "\n" ;

    return 0;
}
