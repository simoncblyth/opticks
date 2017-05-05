CSG Complement
=================

TODO: Look into and handle when intersecting two complemented solids
------------------------------------------------------------------------

* hmm what happens with an intersect of two complemented solids ?? UNDEFINED BEHAVIOR ??


FIXED ISSUE : complemented solids via normal flip yields unexpected intersections (mirror object)
----------------------------------------------------------------------------------------------------

* fixed by special casing complement solid MISS, changing the MISS to an EXIT ...
  as it is impossible to MISS the unbounded complementd solid


* see tboolean-complement for testing this


csg_intersect_part.h::

    164     if(complement)  // flip normal, even for miss need to signal the complement with a -0.f  
    165     {
    166         // For valid_intersects this flips the normal
    167         // otherwise for misses all tt.xyz values should be zero
    168         // but nevertheless proceed to set signbits to signal a complement miss  
    169         // to the caller... csg_intersect_boolean
    170 
    171         tt.x = -tt.x ;
    172         tt.y = -tt.y ;
    173         tt.z = -tt.z ;
    174     }
    175 
    176 }

csg_intersect_boolean.h::

     709                 IntersectionState_t l_state = CSG_CLASSIFY( csg.data[left], ray.direction, tmin );
     710                 IntersectionState_t r_state = CSG_CLASSIFY( csg.data[right], ray.direction, tmin );
     711        
     712 
     713                 float t_left  = fabsf( csg.data[left].w );
     714                 float t_right = fabsf( csg.data[right].w );
     715 
     716                 bool leftIsCloser = t_left <= t_right ;
     717        
     718 #define WITH_COMPLEMENT 1
     719 #ifdef WITH_COMPLEMENT
     720                 // complements (signalled by -0.f) cannot Miss, only Exit, see opticks/notes/issues/csg_complement.rst 
     721 
     722                 // these are only valid (and only needed) for misses 
     723                 bool l_complement = signbit(csg.data[left].x) ;
     724                 bool r_complement = signbit(csg.data[right].x) ;
     725            
     726                 bool l_complement_miss = l_state == State_Miss && l_complement ;              
     727                 bool r_complement_miss = r_state == State_Miss && r_complement ;              
     728            
     729                 if(r_complement_miss)
     730                 {
     731                     r_state = State_Exit ; 
     732                     leftIsCloser = true ; 
     733                 }
     734 
     735                 if(l_complement_miss)
     736                 {
     737                     l_state = State_Exit ; 
     738                     leftIsCloser = false ;
     739                 } 
     740 
     741 #endif     
     742                 int ctrl = boolean_ctrl_packed_lookup( typecode , l_state, r_state, leftIsCloser ) ;
     743                 history_append( hist, nodeIdx, ctrl );




Background
-----------

* Translating DYB Near site geometry yields 22/249 excessively deep(greater that height 3) CSG trees


* tree modification mostly requires positive form (ie with no subtractions, only intersect and union operators
  which are easier to handle as they are commutative)

* making positive trees requires applying De Morgan's laws which require complements



CSG Single Hit Ray Trace Sub-object combination tables
----------------------------------------------------------


* http://xrt.wikidot.com/doc:csg

=============  ==========================  ==========================  =============
UNION            Enter B                    Exit B                      Miss B
=============  ==========================  ==========================  =============
Enter A         ReturnAIfCloser,            ReturnBIfCloser,            ReturnA
                ReturnBIfCloser             AdvanceAAndLoop    
-------------  --------------------------  --------------------------  -------------
Exit A          ReturnAIfCloser,            ReturnAIfFarther,           ReturnA
                AdvanceBAndLoop             ReturnBIfFarther     
-------------  --------------------------  --------------------------  -------------
Miss A          ReturnB                     ReturnB                     ReturnMiss
=============  ==========================  ==========================  =============




=============  ==========================  ==========================  =============
DIFFERENCE      Enter B                     Exit B                      Miss B
=============  ==========================  ==========================  =============
Enter A         ReturnAIfCloser,            AdvanceAAndLoopIfCloser,    ReturnA
                AdvanceBAndLoop             AdvanceBAndLoopIfCloser    
-------------  --------------------------  --------------------------  -------------
Exit A          ReturnAIfCloser,            ReturnBIfCloser,            ReturnA
                                            FlipB
                ReturnBIfCloser,            AdvanceAAndLoop     
                FlipB
-------------  --------------------------  --------------------------  -------------
Miss A          ReturnMiss                  ReturnMiss                  ReturnMiss
=============  ==========================  ==========================  =============



=============  ==========================  ==========================  =============
INTERSECTION    Enter B                     Exit B                      Miss B
=============  ==========================  ==========================  =============
Enter A         AdvanceAAndLoopIfCloser,    ReturnAIfCloser,            ReturnMiss
                AdvanceBAndLoopIfCloser     AdvanceBAndLoop
-------------  --------------------------  --------------------------  -------------
Exit A          ReturnBIfCloser,            ReturnAIfCloser,            ReturnMiss
                AdvanceAAndLoop             ReturnBIfCloser
-------------  --------------------------  --------------------------  -------------
Miss A          ReturnMiss                  ReturnMiss                  ReturnMiss
=============  ==========================  ==========================  =============



Difference is Equivalent to intersect with complement
--------------------------------------------------------


Logical identity::

    A - B  = A INTERSECT !B

    DIFFERENCE(A,B)  = INTERSECT(A,!B)



can single hit CSG implementation handle complements ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* **it must be possible, as it can handle difference** 

* BUT: previous attempts to use unbounded CSG sub-objects (like infinite half-spaces defined by planes) 
  failed  ... the single-hit-CSG algorithm relies on intersecting with the "other" side of sub-objects 

* so long as intersects at infinity are shrouded by "ReturnTheOtherIfCloser" might 
  manage to get away with unbounded ?


* Enter/Exit classification comes from comparison of normal and ray directions


From within !B:

* intersects at infinity will be Exit(!B)
* close intersects (with the bubble) will also be Exit(!B) 
* ... seems no possibility to miss !B ?

* PERHAPS FOR COMPLEMENT-B NEED TO RECLASSIFY, MISS-B -> EXIT-B ?
  MISS-B means it didnt intersect with the local bubble but when its a complement, the
  unbounded nature of !B converts that into EXIT-B ?
  

* for DIFFERENCE(A,B) MISS-B -> ReturnA, need to get the INTERSECT(A,-B) table to ReturnA, 
  intersects at infinity are always going to be further... 

* INTERSECT(A,B) ExitB column bother EnterA,ExitA shrouded by ReturnAIfCloser
  which will always be true.... HENCE SEEMS THAT IT WILL WORK ... 

  * FOR COMPLEMENTS RE-CLASSIFY MISS TO EXIT



INTERSECT(A, !B)
~~~~~~~~~~~~~~~~~~~~~~

Transposes "EnterB" with "ExitB",  getting close to DIFFERENCE(A,B) table

Mismatches being:

* presence of "FlipB" together with both "ReturnBIfCloser" in DIFFERENCE
  (but B is already flipped, so not a difference ?)
 
* presence of "ReturnA" in the "MissB" column of DIFFERENCE vs "ReturnMiss" in INTERSECTION(A,!B)



What does (MISS !B) mean ...
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Think of an inside out sphere (air bubble in water), in some sense there is no way to miss !B, 

* from inside the bubble (outside !B)... close intersect is inevitable
* from outside the bubble (inside !B) 

  * either close intersect with the !B bubble surface
  * OR intersect at infinity with surface of !B 





oxrap/cu/csg_intersect_boolean.h::

     267 #define CSG_CLASSIFY( ise, dir, tmin )   (fabsf((ise).w) > (tmin) ?  ( (ise).x*(dir).x + (ise).y*(dir).y + (ise).z*(dir).z < 0.f ? State_Enter : State_Exit ) : State_Miss )
     ...
     705                 int left  = firstLeft ? csg.curr   : csg.curr-1 ;
     706                 int right = firstLeft ? csg.curr-1 : csg.curr   ;
     707 
     708                 IntersectionState_t l_state = CSG_CLASSIFY( csg.data[left], ray.direction, tmin );
     709                 IntersectionState_t r_state = CSG_CLASSIFY( csg.data[right], ray.direction, tmin );
     710 
     711                 float t_left  = fabsf( csg.data[left].w );
     712                 float t_right = fabsf( csg.data[right].w );
     713 
     714                 int ctrl = boolean_ctrl_packed_lookup( typecode , l_state, r_state, t_left <= t_right ) ;
     715                 history_append( hist, nodeIdx, ctrl );
     716 




