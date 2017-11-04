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


    //  g4-cls SystemOfUnits
    //              mm = 1
    //               m = 1e3
    //              ns = nanosecond = 1 
    //               s = second = 1e9
    //            
    //  g4-cls PhysicalConstants
    //  
    //         c_light   = 2.99792458e+8 * m/s;
    //                     2.99792458e+8 * 1e3/1e9  = 299.79245800000001
    //        
    /*

delta:opticks blyth$ GConstantTest
2017-11-04 18:52:12.837 INFO  [2903258] [main@9] GConstant::meter        1000.000000
2017-11-04 18:52:12.837 INFO  [2903258] [main@10] GConstant::second       1000000000.000000
2017-11-04 18:52:12.837 INFO  [2903258] [main@11] GConstant::electronvolt 0.000001
2017-11-04 18:52:12.837 INFO  [2903258] [main@12] GConstant::nanometer    0.000001
2017-11-04 18:52:12.837 INFO  [2903258] [main@13] GConstant::e_SI         0.000000
2017-11-04 18:52:12.837 INFO  [2903258] [main@14] GConstant::joule        6241509647120.416992
2017-11-04 18:52:12.837 INFO  [2903258] [main@15] GConstant::h_Planck     0.000000
2017-11-04 18:52:12.837 INFO  [2903258] [main@16] GConstant::c_light      299.792458
2017-11-04 18:52:12.837 INFO  [2903258] [main@17] GConstant::hc_eVnm      1239.841875





(lldb) p c_light 
(const double) $30 = 299.79245800000001

(lldb) p h_Planck*c_light*1e12
(double) $38 = 1239.8418754199977


    */


    return 0 ;
}
