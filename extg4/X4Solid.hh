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

#pragma once

#include <vector>
#include <set>
#include "plog/Severity.h"
#include "X4_API_EXPORT.hh"
#include "OpticksCSG.h"
#include "G4ThreeVector.hh"
#include "G4RotationMatrix.hh"
#include "X4SolidBase.hh"
class G4VSolid ; 
class G4BooleanSolid ; 
class G4PolyconeHistorical ; 
class Opticks ; 
struct nnode ; 
struct nmat4triple ; 

/**
X4Solid
==========

Converts G4VSolid into OpticksCSG nnode trees, the number
of nodes in the tree depends on G4VSolid parameter values, 
eg whether an inner radius greater than zero is set, or phi 
segments are set.

Whilst converting G4Solid to nnode the g4code to instanciate
the solids are collected allowing generation of C++ source 
code to create the G4 solids. This g4code is tacked onto the nnode.  
TODO: see if the g4code survives tree balancing, probably cases 
without one-to-one model match will not work. 

Note that cfg4/CMaker can do the opposite conversion::

    G4VSolid* CMaker::MakeSolid(const nnode*)

Contrary to my recollection CMaker is rather complete 
unlike a primordial class of the same name.

NB the results of X4Solid conversions are **not visible in the glTF**
renders, as those are based on the G4Polyhedron polgonization 
of the solids.  Thus the skipping of G4DisplacedSolid displacement
info will impact the ray trace (and the simulation) but not the glTF.

No tree balancing is implemented yet (see ../analytic/csg.py), 
however polycone primitives are hung on a UnionTree and 
the tree is pruned a bit using NTreeBuilder.

TODO
-----

* provide digest methods for each of the ~11 converted G4VSolid, 
  so the geometry digest will notice changes to the solids

**/

class X4_API X4Solid : public X4SolidBase 
{
    struct zplane 
    {
        double rmin ;  
        double rmax ;  
        double z ; 
    };
    public:
        static const plog::Severity  LEVEL ; 
        static void SetVerbosity(unsigned verbosity);
        static bool Contains( const char* s , char c ); 
        static void Banner( int lvIdx, int soIdx, const char* lvname, const char* soname ); 
        static nnode* Convert(const G4VSolid* solid, const Opticks* ok, const char* boundary=NULL, int lvIdx=-1 );
        static nnode* Balance(nnode* raw, int soIdx=-1 , int lvIdx=-1 );  // cannot be const due to inplace positivization
    public:
        X4Solid(const G4VSolid* solid, const Opticks* ok, bool top, int lvIdx=-1); 
    public:
    public:
        bool hasDisplaced() const ; 
        X4Solid* getDisplaced() const ; 
        void setDisplaced(X4Solid* displaced);
    private:
        void init();
        static unsigned fVerbosity ; 
    private:
        void convertDisplacedSolid();
        static void DumpTransform( const char* msg, const nmat4triple* transform ); 
        void convertUnionSolid();
        void convertIntersectionSolid();
        void convertSubtractionSolid();
    private:
        void convertMultiUnion();
        void changeToListSolid(unsigned hint); 
        void convertBooleanSolid();
        static OpticksCSG_t GetOperator( const G4BooleanSolid* solid ); 


        void convertSphere();
        static const bool convertSphere_enable_phi_segment ;  
 
        void convertOrb();
        void convertBox();
        void convertTubs();
        void convertTrd();
        void convertCons();
        void convertTorus();
        void convertEllipsoid();

    private:
        static const bool convertPolycone_enable_phi_segment ; 
        static const int  convertPolycone_debug_mode ;  // export X4Solid_convertPolycone_debug_mode=1 

        void convertPolycone();
        void convertPolycone_g4code();

        static void Polycone_GetZPlane(std::vector<zplane>& zp, std::set<double>& R_inner, std::set<double>& R_outer, const G4PolyconeHistorical* ph  ); 
        static void Polycone_MakePrims( const std::vector<zplane>& zp,  std::vector<nnode*>& prims, const char* name, bool outer  ); 
        static nnode* Polycone_MakeInner(const std::vector<zplane>& zp, const char* name, unsigned num_R_inner); 
        static bool Polycone_DoPhiSegment( const G4PolyconeHistorical* ph ); 
        static bool Polycone_CheckZOrder( const std::vector<zplane>& zp, bool z_ascending ); 
        static void SetExternalBoundingBox( nnode* root,  const G4VSolid* solid ); 
    private:
        void convertHype();
    private:
        static const float hz_disc_cylinder_cut ; 
        nnode* convertTubs_disc();
        nnode* convertTubs_cylinder(bool do_nudge_inner);
    private:
        nnode* intersectWithPhiSegment(nnode* whole, float startPhi, float deltaPhi, float segZ, float segR );
        static const int intersectWithPhiSegment_debug_mode ; 
        void booleanDisplacement( G4VSolid** pp, G4ThreeVector& pos, G4ThreeVector& rot );
        G4ThreeVector GetAngles(const G4RotationMatrix& mtx);
    private:
        nnode* intersectWithPhiCut(  nnode* whole, double startPhi_pi   , double deltaPhi_pi   ); 
        nnode* intersectWithThetaCut(nnode* whole, double startTheta_pi , double deltaTheta_pi ); 
    private:
        nnode* convertSphere_(bool only_inner);
        nnode* convertSphereDEV_(const char* opt);
    private:
        nnode* convertCons_(bool only_inner);
        nnode* convertHype_(bool only_inner);

    private:
        X4Solid* m_displaced ; 

};

