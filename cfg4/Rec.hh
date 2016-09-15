#pragma once

#include "G4OpBoundaryProcess.hh"
#include <vector>

class CPropLib ; 
class Opticks ; 
class OpticksEvent ; 
class State ; 


/*

Mapping G4Step/G4StepPoint into Opticks records style is the point of *Recorder*


Truncation that matches optixrap-/cu/generate.cu::

    generate...

    int bounce = 0 ;
    int slot = 0 ;
    int slot_min = photon_id*MAXREC ;       // eg 0 for photon_id=0
    int slot_max = slot_min + MAXREC - 1 ;  // eg 9 for photon_id=0, MAXREC=10
    int slot_offset = 0 ;

    while( bounce < bounce_max )
    {
        bounce++

        rtTrace...

        slot_offset =  slot < MAXREC  ? slot_min + slot : slot_max ;

          // eg 0,1,2,3,4,5,6,7,8,9,9,9,9,9,....  if bounce_max were greater than MAXREC
          //    0,1,2,3,4,5,6,7,8,9       for bounce_max = 9, MAXREC = 10 

        RSAVE(..., slot, slot_offset)...
        slot++ ;

        propagate_to_boundary...
        propagate_at_boundary... 
    }

    slot_offset =  slot < MAXREC  ? slot_min + slot : slot_max ;

    RSAVE(..., slot, slot_offset)


Consider truncated case with bounce_max = 9, MAXREC = 10 

* last while loop starts at bounce = 8 
* RSAVE inside the loop invoked with bounce=1:9 
  and then once more beyond the while 
  for a total of 10 RSAVEs 


 Opticks records...

       flag set to generation code
       while(bounce < bounce_max)
       {
             rtTrace(..)     // look ahead to get boundary 

             fill_state()    // interpret boundary into m1 m2 material codes based on photon direction

             RSAVE()         

                             // change photon position/direction/time/flag 
             propagate_to_boundary
             propagate_at_surface or at_boundary   

             break/continue depending on flag
       }

       RSAVE()

  Consider reflection

     G4:
         G4Step at boundary straddles the volumes, with post step point 
         in a volume that will not be entered. 

         Schematic of the steps and points of a reflection


        step1    step2    step3


          *      .   .
           \     .   .
            \    .   .
             \   .   . 
              \  .   .  
               * . * .
                 . * .  *
                 .   .   \
                 .   .    \
                 .   .     \
                 .   .      * 
                 .   . 


       At each step the former *post* becomes the *pre*.
       So just taking the *pre* will get all points, if special case the 
       last step to get *post* too (as no next step).

       At a boundary (eg for BT or BR) the *pre* and *post* are exactly 
       the same except the volume/material assigned. So need to skip to 
       avoid repeating a point.


     Op:
         m2 gets set to the material that will not be entered by the lookahead
         m1 always stays the material that photon is in 

*/




#include "CFG4_API_EXPORT.hh"
#include "CFG4_HEAD.hh"

class CFG4_API Rec {
   public:
       typedef enum { OK, SKIP_STS } Rec_t ; 
       typedef enum { PRE, POST } Flag_t ; 
   public:
       Rec(Opticks* ok, CPropLib* clib, bool dynamic);
       void initEvent(OpticksEvent* evt);
   private:
       void setEvent(OpticksEvent* evt);
   public:
       void add(const State* state); 
       void sequence();
       void Clear();
   public:
       void Dump(const char* msg); 
   public:
       G4OpBoundaryProcessStatus getBoundaryStatus(unsigned int i);
       const State* getState(unsigned int i);
       unsigned int getNumStates();
   public:
       Rec_t getFlagMaterial(unsigned int& flag, unsigned int& material, unsigned int i, Flag_t type );
   public:
       void addFlagMaterial(unsigned int flag, unsigned int material);
       unsigned long long getSeqHis();
       unsigned long long getSeqMat();

       void setDebug(bool debug=true);
   private:
       Opticks*                    m_ok ;  
       CPropLib*                   m_clib ; 
       bool                        m_dynamic ; 

       OpticksEvent*               m_evt ;  
       unsigned int                m_genflag ;
       std::vector<const State*>   m_states ; 

       unsigned long long          m_seqhis ; 
       unsigned long long          m_seqmat ; 
       unsigned int                m_slot ; 

       unsigned int m_record_max ; 
       unsigned int m_bounce_max ; 
       unsigned int m_steps_per_photon ; 

       bool         m_debug ; 
};

#include "CFG4_TAIL.hh"

