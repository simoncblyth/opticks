full_geom_missing_ellipsoid_transform_again
=============================================


::

    829 G4VPhysicalVolume* G4GDMLReadStructure::
    830 GetPhysvol(const G4String& ref) const
    831 {
    832    G4VPhysicalVolume* physvolPtr =
    833      G4PhysicalVolumeStore::GetInstance()->GetVolume(ref,false);
    834 
    835    if (!physvolPtr)
    836    {
    837      G4String error_msg = "Referenced physvol '" + ref + "' was not found!";
    838      G4Exception("G4GDMLReadStructure::GetPhysvol()", "ReadError",
    839                  FatalException, error_msg);
    840    }
    841 
    842    return physvolPtr;
    843 }


    062 class G4PhysicalVolumeStore : public std::vector<G4VPhysicalVolume*>

    160 // ***************************************************************************
    161 // Retrieve the first volume pointer in the container having that name
    162 // ***************************************************************************
    163 //
    164 G4VPhysicalVolume*
    165 G4PhysicalVolumeStore::GetVolume(const G4String& name, G4bool verbose) const
    166 {
    167   for (iterator i=GetInstance()->begin(); i!=GetInstance()->end(); i++)
    168   {
    169     if ((*i)->GetName() == name) { return *i; }
    170   }
    171   if (verbose)
    172   {
    173      std::ostringstream message;
    174      message << "Volume NOT found in store !" << G4endl
    175             << "        Volume " << name << " NOT found in store !" << G4endl
    176             << "        Returning NULL pointer.";
    177      G4Exception("G4PhysicalVolumeStore::GetVolume()",
    178                  "GeomMgt1001", JustWarning, message);
    179   }
    180   return 0;
    181 }





    845 G4LogicalVolume* G4GDMLReadStructure::
    846 GetVolume(const G4String& ref) const
    847 {
    848    G4LogicalVolume *volumePtr
    849    = G4LogicalVolumeStore::GetInstance()->GetVolume(ref,false);
    850 
    851    if (!volumePtr)
    852    {
    853      G4String error_msg = "Referenced volume '" + ref + "' was not found!";
    854      G4Exception("G4GDMLReadStructure::GetVolume()", "ReadError",
    855                  FatalException, error_msg);
    856    }
    857 
    858    return volumePtr;
    859 }


::

   062 class G4LogicalVolumeStore : public std::vector<G4LogicalVolume*>

   0155 // ***************************************************************************
    156 // Retrieve the first volume pointer in the container having that name
    157 // ***************************************************************************
    158 //
    159 G4LogicalVolume*
    160 G4LogicalVolumeStore::GetVolume(const G4String& name, G4bool verbose) const
    161 {
    162   for (iterator i=GetInstance()->begin(); i!=GetInstance()->end(); i++)
    163   {
    164     if ((*i)->GetName() == name) { return *i; }
    165   }
    166   if (verbose)
    167   {
    168      std::ostringstream message;
    169      message << "Volume NOT found in store !" << G4endl
    170              << "        Volume " << name << " NOT found in store !" << G4endl
    171              << "        Returning NULL pointer.";
    172      G4Exception("G4LogicalVolumeStore::GetVolume()",
    173                  "GeomMgt1001", JustWarning, message);
    174   }
    175   return 0;
    176 }

    182 G4LogicalVolumeStore* G4LogicalVolumeStore::GetInstance()
    183 {
    184   static G4LogicalVolumeStore worldStore;
    185   if (!fgInstance)
    186   {
    187     fgInstance = &worldStore;
    188   }
    189   return fgInstance;
    190 }






Check PLS.txt from two runs::

     212895 solidXJfixture0x595eb40
     212896 pLPMT_Hamamatsu_R128600x5f67fb0
     212897 HamamatsuR12860lMaskVirtual0x5f51160
     212898 HamamatsuR12860sMask_virtual0x5f50520
     212899 HamamatsuR12860pMask0x5f51db0
     212900 HamamatsuR12860lMask0x5f51c50
     212901 HamamatsuR12860sMask0x5f51a40
     212902 HamamatsuR12860pMaskTail0x5f53210


     212895 solidXJfixture0x595eb40
     212896 pLPMT_Hamamatsu_R128600x5f67fb0
     212897 HamamatsuR12860lMaskVirtual0x5f51160
     212898 HamamatsuR12860sMask_virtual0x5f50520
     212899 HamamatsuR12860pMask0x5f51db0
     212900 HamamatsuR12860lMask0x5f51c50
     212901 HamamatsuR12860sMask0x5f51a40
     212902 HamamatsuR12860pMaskTail0x5f53210
     212903 HamamatsuR12860lMaskTail0x5f530c0
     212904 HamamatsuR12860Tail0x5f52eb0




