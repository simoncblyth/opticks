#include "SSim.hh"
#include "NP.hh"
#include "SOpticksResource.hh"
#include "OPTICKS_LOG.hh"


void test_Load()
{
    SSim* sim = SSim::Load(); 
    std::cout << sim->desc() ;  
}

void test_findName()
{
    std::vector<std::string> names = {
        "Air", 
        "Rock", 
        "Water", 
        "Acrylic",
        "Cream", 
        "vetoWater", 
        "Cheese", 
        "NextIsBlank",
        "",
        "Galactic", 
        "Pyrex", 
        "PMT_3inch_absorb_logsurf1", 
        "Steel", 
        "Steel_surface",
        "PE_PA",
        "Candy",
        "NextIsBlank",
        "",
        "Steel"
      } ; 


    const SSim* sim = SSim::Load(); 
    const NP* bnd = sim->get(SSim::BND); 
    std::cout << " bnd " << bnd->sstr() << std::endl ; 
    std::cout << " bnd.names " << bnd->names.size() << std::endl ; 


    for(unsigned a=0 ; a < names.size() ; a++ )
    {
         const std::string& n = names[a] ; 
         unsigned i, j ; 
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


void test_addFake()
{
    SSim* sim = SSim::Load(); 
    std::string bef = sim->desc(); 

    LOG(info) << "BEFORE " << std::endl << sim->descOptical() << std::endl ; 

    std::vector<std::string> specs = { "Rock/perfectAbsorbSurface/perfectAbsorbSurface/Air", "Air///Water" } ;
    sim->addFake_(specs); 
    std::string aft = sim->desc(); 

    LOG(info) << "AFTER " << std::endl << sim->descOptical() << std::endl ; 

    std::cout << "bef" << std::endl << bef << std::endl ; 
    std::cout << "aft" << std::endl << aft << std::endl ; 
}

void test_addFake_ellipsis()
{
    SSim* sim = SSim::Load(); 
    sim->addFake("Air///Water"); 
    sim->addFake("Air///Water", "Rock/perfectAbsorbSurface/perfectAbsorbSurface/Air"); 
    sim->addFake("Air///Water", "Rock/perfectAbsorbSurface/perfectAbsorbSurface/Air", "Water///Air" ); 

}



int main(int argc, char** argv)
{ 
    OPTICKS_LOG(argc, argv); 
   
    /*
    test_Load(); 
    test_findName(); 
    test_addFake();     
    */
    test_addFake_ellipsis();     


    return 0 ; 
}



