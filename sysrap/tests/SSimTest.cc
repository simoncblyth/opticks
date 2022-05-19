#include "SSim.hh"
#include "NP.hh"
#include "SOpticksResource.hh"
#include "OPTICKS_LOG.hh"


void test_Add()
{
    const char* cfbase = SOpticksResource::CFBase() ; 
    LOG(info) << " cfbase " << cfbase ; 
    NP* optical = NP::Load(cfbase, "CSGFoundry", "optical.npy"); 
    NP* bnd     = NP::Load(cfbase, "CSGFoundry", "bnd.npy"); 

    LOG(info) << "BEFORE " << std::endl << SSim::DescOptical(optical, bnd ) << std::endl ; 

    NP* opticalplus = nullptr ; 
    NP* bndplus = nullptr ; 
    std::vector<std::string> specs = { "Rock/perfectAbsorbSurface/perfectAbsorbSurface/Air", "Air///Water" } ;

    SSim::Add( &opticalplus, &bndplus, optical, bnd, specs ); 

    LOG(info) << "AFTER " << std::endl << SSim::DescOptical(opticalplus, bndplus ) << std::endl ; 
}

void test_findName(const SSim* sim)
{
    std::vector<std::string> names = {
        "Air", 
        "Rock", 
        "Water", 
        "Acrylic",
        "Cream", 
        "vetoWater", 
        "Cheese", 
        "",
        "Galactic", 
        "Pyrex", 
        "PMT_3inch_absorb_logsurf1", 
        "Steel", 
        "Steel_surface",
        "PE_PA",
        "Candy",
        ""
      } ; 

    unsigned i, j ; 

    const NP* bnd = sim->get(SSim::BND); 

    for(unsigned a=0 ; a < names.size() ; a++ )
    {
         const std::string& n = names[a] ; 
         bool found = sim->findName(i,j,n.c_str() ); 

         std::cout << std::setw(30) << n << " " ; 
         if(found)  
         {
            std::cout 
                << "(" 
                << std::setw(3) << i 
                << ","  
                << std::setw(3) << j
                << ")"
                << " "
                << SSim::GetItemDigest(bnd, i, j, 8)
                ;
         }
         else
         {
            std::cout << "-" ;  
         }
         std::cout << std::endl ;  
    }
}




int main(int argc, char** argv)
{ 
    OPTICKS_LOG(argc, argv); 

    const char* cfbase = SOpticksResource::CFBase("CFBASE") ; 
    LOG(info) << " cfbase " << cfbase ; 
    NP* bnd = NP::Load(cfbase, "CSGFoundry", "bnd.npy"); 

    SSim* sim = SSim::Get(); 
    sim->add(SSim::BND, bnd); 
 
    //test_Add();     
    test_findName(sim); 


    return 0 ; 
}



