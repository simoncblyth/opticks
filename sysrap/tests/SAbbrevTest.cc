// TEST=SAbbrevTest om-t

#include <string>
#include <vector>

#include "SAbbrev.hh"

#include "OPTICKS_LOG.hh"


int main(int argc, char** argv )
{
    std::vector<std::string> ss = { 
        "Acrylic",
        "Air", 
        "Aluminium",
        "Bialkali",
        "DeadWater",
        "ESR",
        "Foam",
        "GdDopedLS",
        "IwsWater",
        "LiquidScintillator",
        "MineralOil",
        "Nitrogen",
        "NitrogenGas",
        "Nylon",
        "OwsWater",
        "PPE",
        "PVC",
        "Pyrex",
        "Rock",
        "StainlessSteel",
        "Tyvek",
        "UnstStainlessSteel",
        "Vacuum",
        "OpaqueVacuum",
        "Water",
        "GlassSchottF2"
    } ;

    SAbbrev ab(ss);
    ab.dump(); 

    return 0 ; 
}

/*
        "ADTableStainlessSteel": "AS",
        "Acrylic": "Ac",
        "Air": "Ai",
        "Aluminium": "Al",
        "Bialkali": "Bk",
        "DeadWater": "Dw",
        "ESR": "ES",
        "Foam": "Fo",
        "GdDopedLS": "Gd",
        "IwsWater": "Iw",
        "LiquidScintillator": "LS",
        "MineralOil": "MO",
        "Nitrogen": "Ni",
        "NitrogenGas": "NG",
        "Nylon": "Ny",
        "OwsWater": "Ow",
        "PPE": "PP",
        "PVC": "PV",
        "Pyrex": "Py",
        "Rock": "Rk",
        "StainlessSteel": "SS",
        "Tyvek": "Ty",
        "UnstStainlessSteel": "US",
        "Vacuum": "Vm",
        "OpaqueVacuum": "OV",
        "Water": "Wt",
        "GlassSchottF2": "F2"
*/

