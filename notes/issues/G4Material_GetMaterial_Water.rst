G4Material_GetMaterial_Water.rst
=====================================


Zike reports some noise::

    Hi Simon,

    Opticks log outputs:
    .....
    G4Material::GetMaterial() WARNING: The material: Water does not exist in the table. Return NULL pointer.
    G4Material::GetMaterial() WARNING: The material: Water does not exist in the table. Return NULL pointer.
    .....
    2022-11-24 15:09:32.454 ERROR [3092411] [QSim::UploadComponents@156]  icdf null, SSim::ICDF icdf.npy
    .....

What do these means? In my simulation, there should be no material called
"Water". The material of water I used is manually created and called
"Water_4degree".::

    epsilon:opticks blyth$ opticks-f "G4Material::GetMaterial(\"Water\")"  
    ./extg4/X4MaterialWater.cc:    G4Material* Water = G4Material::GetMaterial("Water");
    ./extg4/X4MaterialWater.cc:    Water(G4Material::GetMaterial("Water")),
    ./extg4/X4OpRayleigh.cc:    Water(G4Material::GetMaterial("Water")),
    ./u4/U4VolumeMaker.cc:    G4Material* water_material  = G4Material::GetMaterial("Water");   assert(water_material); 

Added warn=false to avoid the noise, if it is a serious lack there will be subseqent assert::

    epsilon:opticks blyth$ opticks-f "G4Material::GetMaterial(\"Water" 
    ./extg4/X4MaterialWater.cc:    G4Material* Water = G4Material::GetMaterial("Water", warn );
    ./extg4/X4MaterialWater.cc:    Water(G4Material::GetMaterial("Water", warn)),
    ./extg4/X4OpRayleigh.cc:    Water(G4Material::GetMaterial("Water", warn)),
    ./u4/U4VolumeMaker.cc:    G4Material* water_material  = G4Material::GetMaterial("Water", warn);   assert(water_material); 
    epsilon:opticks blyth$ 


