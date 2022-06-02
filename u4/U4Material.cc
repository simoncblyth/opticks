#include "G4Material.hh"
#include "U4Material.hh"

G4Material* U4Material::Get(const char* name)
{
   G4Material* material = G4Material::GetMaterial(name); 
   if( material == nullptr )
   {   
       material = Get_(name); 
   }   
   return material ;   
}


G4Material* U4Material::Get_(const char* name)
{
   G4Material* material = nullptr ; 
   if(strcmp(name, "Vacuum")==0)  material = Vacuum(name); 
   return material ; 
}

G4Material* U4Material::Vacuum(const char* name)
{
    G4double z, a, density ;
    G4Material* material = new G4Material(name, z=1., a=1.01*CLHEP::g/CLHEP::mole, density=CLHEP::universe_mean_density );
    return material ;
}


