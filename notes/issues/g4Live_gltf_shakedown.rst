g4Live_gltf_shakedown
========================

glTF viz shows messed up transforms
--------------------------------------

Debug by editing the glTF to pick particular nodes::

    329578   "scenes": [
    329579     {
    329580       "nodes": [
    329581         3199
    329582       ]
    329583     }


::

   3199 : single pmt (with frame false looks correct, with frame true mangled)
   3155 : AD  (view starts from above the lid) (with frame false PMT all pointing in one direction, with frame true correct)
   3147 : pool with 2 ADs etc..


Similar trouble before
------------------------


Every time, gets troubles from transforms...

* :doc:`gdml_gltf_transforms`



Debugging Approach ?
-----------------------

* compare the GGeo transforms from the two streams 
* simplify transform handling : avoid multiple holdings of transforms, 
  
Observations

* assembly of the PMT within its "frame" (of 5 parts) only involves 
  translation in z : so getting that correct could be deceptive as no rotation   



Switching to frame gets PMT pointing correct, but seems mangled inside themselves
-----------------------------------------------------------------------------------

* mangled : the base poking thru the front 


::

     20 glm::mat4* X4Transform3D::GetLocalTransform(const G4VPhysicalVolume* const pv, bool frame)
     21 {    
     22     glm::mat4* transform = NULL ;
     23     if(frame)
     24     {
     25         const G4RotationMatrix* rotp = pv->GetFrameRotation() ;
     26         G4ThreeVector    tla = pv->GetFrameTranslation() ;
     27         G4Transform3D    tra(rotp ? *rotp : G4RotationMatrix(),tla);
     28         transform = new glm::mat4(Convert( tra ));
     29     }   
     30     else
     31     {
     32         G4RotationMatrix rot = pv->GetObjectRotationValue() ;  // obj relative to mother
     33         G4ThreeVector    tla = pv->GetObjectTranslation() ; 
     34         G4Transform3D    tra(rot,tla);
     35         transform = new glm::mat4(Convert( tra ));
     36     }   
     37     return transform ;
     38 }   




FIXED : bad mesh association, missing meshes
------------------------------------------------

Also add metadata extras to allow to navigate the gltf.  Suspect 
are getting bad mesh association, as unexpected lots of repeated mesh.

Huh : only 35 meshes, (expect ~250) but the expected 12k nodes.

Suspect the lvIdx mesh identity.




