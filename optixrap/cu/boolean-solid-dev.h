/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */



/*
Perhaps could treat the LUT like a matrix and get into optix that way 

rtDeclareVariable(optix::Matrix4x4, textureMatrix0, , );

* https://devtalk.nvidia.com/default/topic/739954/optix/emulating-opengl-texture-matrix/
* https://devtalk.nvidia.com/default/topic/767650/optix/matrix4x4-assignment-not-working/

Or as tiny 3D buffer/texture 
*/

#ifdef __CUDACC__
__host__
__device__
#endif
int boolean_lookup( OpticksCSG_t op, IntersectionState_t stateA, IntersectionState_t stateB )
{
    //
    // assumes OpticksCSG_t e-n-u-m values 0,1,2 for CSG_UNION, CSG_INTERSECTION, CSG_DIFFERENCE
    // and IntersectionState_t e-n-u-m values 0,1,2 for Enter, Exit, Miss
    // split up e--n--u--m for pytin 
    //
    // cannot use static here with CUDA, does that mean the lookup gets created every time ?
    //
    const unsigned _boolean_lookup[3][3][3] = 
          { 
             { 
                 {Union_EnterA_EnterB, Union_EnterA_ExitB, Union_EnterA_MissB },
                 {Union_ExitA_EnterB,  Union_ExitA_ExitB,  Union_ExitA_MissB },
                 {Union_MissA_EnterB,  Union_MissA_ExitB,  Union_MissA_MissB }
             },
             { 
                 {Intersection_EnterA_EnterB, Intersection_EnterA_ExitB, Intersection_EnterA_MissB },
                 {Intersection_ExitA_EnterB,  Intersection_ExitA_ExitB,  Intersection_ExitA_MissB },
                 {Intersection_MissA_EnterB,  Intersection_MissA_ExitB,  Intersection_MissA_MissB }
             },
             { 
                 {Difference_EnterA_EnterB, Difference_EnterA_ExitB, Difference_EnterA_MissB },
                 {Difference_ExitA_EnterB,  Difference_ExitA_ExitB,  Difference_ExitA_MissB },
                 {Difference_MissA_EnterB,  Difference_MissA_ExitB,  Difference_MissA_MissB }
             }
         } ;
    return _boolean_lookup[(int)op][(int)stateA][(int)stateB] ; 
}




#ifdef __CUDACC__
__host__
__device__
#endif
int union_action( IntersectionState_t stateA, IntersectionState_t stateB  )
{
    // NB NO LONGER IN USE
    int offset = 3*(int)stateA + (int)stateB ;   
    int action = Union_MissA_MissB ; 
    switch(offset)
    {
       case 0: action=Union_EnterA_EnterB ; break ; 
       case 1: action=Union_EnterA_ExitB  ; break ; 
       case 2: action=Union_EnterA_MissB  ; break ; 
       case 3: action=Union_ExitA_EnterB  ; break ; 
       case 4: action=Union_ExitA_ExitB   ; break ; 
       case 5: action=Union_ExitA_MissB   ; break ; 
       case 6: action=Union_MissA_EnterB  ; break ; 
       case 7: action=Union_MissA_ExitB   ; break ; 
       case 8: action=Union_MissA_MissB   ; break ; 
    }
    return action ; 
}

#ifdef __CUDACC__
__host__
__device__
#endif
int intersection_action( IntersectionState_t stateA, IntersectionState_t stateB  )
{
    // NB NO LONGER IN USE
    int offset = 3*(int)stateA + (int)stateB ;   
    int action = Intersection_MissA_MissB ; 
    switch(offset)
    {
       case 0: action=Intersection_EnterA_EnterB ; break ; 
       case 1: action=Intersection_EnterA_ExitB  ; break ; 
       case 2: action=Intersection_EnterA_MissB  ; break ; 
       case 3: action=Intersection_ExitA_EnterB  ; break ; 
       case 4: action=Intersection_ExitA_ExitB   ; break ; 
       case 5: action=Intersection_ExitA_MissB   ; break ; 
       case 6: action=Intersection_MissA_EnterB  ; break ; 
       case 7: action=Intersection_MissA_ExitB   ; break ; 
       case 8: action=Intersection_MissA_MissB   ; break ; 
    }
    return action ; 
}

#ifdef __CUDACC__
__host__
__device__
#endif
int difference_action( IntersectionState_t stateA, IntersectionState_t stateB  )
{
    // NB NO LONGER IN USE
    int offset = 3*(int)stateA + (int)stateB ;   
    int action = Difference_MissA_MissB ; 
    switch(offset)
    {
       case 0: action=Difference_EnterA_EnterB ; break ; 
       case 1: action=Difference_EnterA_ExitB  ; break ; 
       case 2: action=Difference_EnterA_MissB  ; break ; 
       case 3: action=Difference_ExitA_EnterB  ; break ; 
       case 4: action=Difference_ExitA_ExitB   ; break ; 
       case 5: action=Difference_ExitA_MissB   ; break ; 
       case 6: action=Difference_MissA_EnterB  ; break ; 
       case 7: action=Difference_MissA_ExitB   ; break ; 
       case 8: action=Difference_MissA_MissB   ; break ; 
    }
    return action ; 
}



#ifdef __CUDACC__
__host__
__device__
#endif

int boolean_actions( OpticksCSG_t operation, IntersectionState_t stateA, IntersectionState_t stateB  )
{
    // NB NO LONGER IN USE
    int action = ReturnMiss ; 
    switch(operation)
    {
       case CSG_INTERSECTION: action = intersection_action( stateA, stateB ) ; break ;
       case CSG_UNION:        action = union_action( stateA, stateB ) ; break ;
       case CSG_DIFFERENCE:   action = difference_action( stateA, stateB ) ; break ;
       case CSG_PRIMITIVE:    action = ReturnMiss                          ; break ;   // perhaps return error flag ?
    }
    return action ; 
}









#ifdef __CUDACC__
__host__
__device__
#endif
int boolean_decision(int acts, bool ACloser)
{
    int act = ReturnMiss ; 

    // convert potentially multiple bits of acts into single bit act 
    // NB the order of this if is critical
    //
    // hmm : not so many possible values of acts and two possible values of ACloser
    //       can this been done via lookup too ?
    //
    // hmm the ordering multiplies the possible "states" for the lookup ?
    //      but not by that much, especially when split into two for ACloser or not
    //

    if      (acts & ReturnMiss)                            act = ReturnMiss ; 
    else if (acts & ReturnA)                               act = ReturnA ; 
    else if ((acts & ReturnAIfCloser) && ACloser)          act = ReturnAIfCloser ; 
    else if ((acts & ReturnAIfFarther) && !ACloser)        act = ReturnAIfFarther ; 
    else if (acts & ReturnB)                               act = ReturnB ;
    else if ((acts & ReturnBIfCloser) && !ACloser)         act = ReturnBIfCloser ;
    else if ((acts & ReturnFlipBIfCloser) && !ACloser)     act = ReturnFlipBIfCloser ;
    else if ((acts & ReturnBIfFarther) && ACloser)         act = ReturnBIfFarther ;
    else if (acts & AdvanceAAndLoop)                       act = AdvanceAAndLoop ;
    else if ((acts & AdvanceAAndLoopIfCloser) && ACloser)  act = AdvanceAAndLoopIfCloser ;
    else if (acts & AdvanceBAndLoop)                       act = AdvanceBAndLoop ;
    else if ((acts & AdvanceBAndLoopIfCloser) && !ACloser) act = AdvanceBAndLoopIfCloser ;

    return act  ;
}


#ifdef __CUDACC__
__host__
__device__
#endif
int boolean_ctrl(int act)
{
    int ctrl = CTRL_UNDEFINED ; 
    if(act & ReturnMiss)                                           ctrl = CTRL_RETURN_MISS ; 
    else if( act & (ReturnA | ReturnAIfCloser | ReturnAIfFarther)) ctrl = CTRL_RETURN_A ; 
    else if( act & (ReturnB | ReturnBIfCloser | ReturnBIfFarther)) ctrl = CTRL_RETURN_B ; 
    else if( act & ReturnFlipBIfCloser)                            ctrl = CTRL_RETURN_FLIP_B ; 
    else if( act & (AdvanceAAndLoop | AdvanceAAndLoopIfCloser))    ctrl = CTRL_LOOP_A  ; 
    else if( act & (AdvanceBAndLoop | AdvanceBAndLoopIfCloser))    ctrl = CTRL_LOOP_B  ; 
    return ctrl ; 
}



