// TEST=NLookupTest om-t

// ggv --lookup
// ggv --jpmt --lookup


#include "NLookup.hpp"
#include "Opticks.hh"
#include "GBndLib.hh"

#include "OPTICKS_LOG.hh"

#include "GGEO_BODY.hh"
#include "PLOG.hh"

/*

ChromaMaterialMap.json contains name to code mappings used 
for a some very old gensteps that were produced by G4DAEChroma
and which are still in use.
As the assumption of all gensteps being produced the same
way and with the same material mappings will soon become 
incorrect, need a more flexible way.

Perhaps a sidecar file to the gensteps .npy should
contain material mappings, and if it doesnt exist then 
defaults are used ?

::

    simon:DayaBay blyth$ cat ChromaMaterialMap.json | tr "," "\n"
    {"/dd/Materials/OpaqueVacuum": 18
     "/dd/Materials/Pyrex": 21
     "/dd/Materials/PVC": 20
     "/dd/Materials/NitrogenGas": 16
     "/dd/Materials/Teflon": 24
     "/dd/Materials/ESR": 9
     "/dd/Materials/MineralOil": 14
     "/dd/Materials/Vacuum": 27
     "/dd/Materials/Bialkali": 5
     "/dd/Materials/Air": 2
     "/dd/Materials/OwsWater": 19
     "/dd/Materials/C_13": 6
     "/dd/Materials/IwsWater": 12
     "/dd/Materials/ADTableStainlessSteel": 0
     "/dd/Materials/Ge_68": 11
     "/dd/Materials/Acrylic": 1
     "/dd/Materials/Tyvek": 25
     "/dd/Materials/Water": 28
     "/dd/Materials/Nylon": 17
     "/dd/Materials/LiquidScintillator": 13
     "/dd/Materials/GdDopedLS": 10
     "/dd/Materials/UnstStainlessSteel": 26
     "/dd/Materials/BPE": 4
     "/dd/Materials/Silver": 22
     "/dd/Materials/DeadWater": 8
     "/dd/Materials/Co_60": 7
     "/dd/Materials/Aluminium": 3
     "/dd/Materials/Nitrogen": 15
     "/dd/Materials/StainlessSteel": 23}

*/








int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);


    // load GBndLib from cache and dump
    Opticks* ok = new Opticks(argc, argv);
    GBndLib* blib = GBndLib::load(ok, true );
    blib->dump();
    blib->dumpMaterialLineMap("GBndLib.MaterialLineMap : texline to material name correspondence");


    // load material name to index mapping from ChromaMaterialMap.json
    // these these indices are used by some old gensteps 
    NLookup* m_lookup = new NLookup();
    const char* cmmd = ok->getDetectorBase() ;
    m_lookup->loadA( cmmd , "ChromaMaterialMap.json", "/dd/Materials/") ;



    const std::map<std::string, unsigned int>& B = blib->getMaterialLineMap() ;
    m_lookup->setB(B,"","NLookupTest/blib");    // shortname eg "GdDopedLS" to material line mapping 

    m_lookup->close("ggeo-/NLookupTest");


    printf("  a => b \n");
    for(unsigned int a=0; a < 35 ; a++ )
    {   
        int b = m_lookup->a2b(a);
        std::string aname = m_lookup->acode2name(a) ;
        std::string bname = m_lookup->bcode2name(b) ;

        int c = b/4 ; 
        int d = b - c*4 ; 

        std::cout 
            << " " 
            << std::setw(3) << a 
            << " -> " 
            << std::setw(3) << b
            << "   "
            << "(" 
            << std::setw(2) << c 
            << "," 
            << std::setw(1) << d 
            <<  ")" 
            ; 


        if(b < 0) 
        {
            LOG(warning) 
                       << " FAILED TO TRANSLATE a->b "
                       << " a " << std::setw(3) << a 
                       << " b " << std::setw(3) << b
                       << " an " << std::setw(25) << aname 
                       << " bn " << std::setw(25) << bname 
                       ;
        } 
        else
        {   
             assert(aname == bname);
             std::cout << " " << aname << std::endl ; 
        }   
    }   
}



