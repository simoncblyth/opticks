#include <cassert>
#include "GConstant.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    LOG(info) << "GConstant::meter        " << std::fixed << std::setprecision(6) << GConstant::meter ;
    LOG(info) << "GConstant::second       " << std::fixed << std::setprecision(6) << GConstant::second ;
    LOG(info) << "GConstant::electronvolt " << std::fixed << std::setprecision(6) << GConstant::electronvolt ;
    LOG(info) << "GConstant::nanometer    " << std::fixed << std::setprecision(6) << GConstant::nanometer ;
    LOG(info) << "GConstant::e_SI         " << std::fixed << std::setprecision(6) << GConstant::e_SI ;
    LOG(info) << "GConstant::joule        " << std::fixed << std::setprecision(6) << GConstant::joule ;
    LOG(info) << "GConstant::h_Planck     " << std::fixed << std::setprecision(6) << GConstant::h_Planck ;
    LOG(info) << "GConstant::c_light      " << std::fixed << std::setprecision(6) << GConstant::c_light ;
    LOG(info) << "GConstant::hc_eVnm      " << std::fixed << std::setprecision(6) << GConstant::hc_eVnm ;

    assert(GConstant::meter == 1000.f ); 


    return 0 ;
}
