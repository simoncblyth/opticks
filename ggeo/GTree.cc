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

#include "NSensor.hpp"
#include "NPY.hpp"

#include "GNode.hh"
#include "GVolume.hh"
#include "GTree.hh"

#include "PLOG.hh"

const plog::Severity GTree::LEVEL = PLOG::EnvLevel("GTree", "DEBUG") ; 


/**
GTree::makeInstanceTransformsBuffer
-------------------------------------

Returns transforms array of shape (num_instances, 4, 4)

Collects transforms from GNode placement instances into a buffer.
getPlacement for ridx=0 just returns m_root (which always has identity transform)
for ridx > 0 returns all GNode subtree bases of the ridx repeats.

**/

NPY<float>* GTree::makeInstanceTransformsBuffer(const std::vector<const GNode*>& placements) // static
{
    unsigned ni = placements.size(); 
    NPY<float>* buf = NPY<float>::make(0, 4, 4);
    for(unsigned int i=0 ; i < ni ; i++)
    {
        const GNode* place = placements[i] ;
        GMatrix<float>* t = place->getTransform();
        buf->add(t->getPointer(), 4*4*sizeof(float) );
    } 
    assert(buf->getNumItems() == ni);
    return buf ; 
}



/**
GTree::makeInstanceIdentityBuffer
-----------------------------------

Canonically invoked by GMergedMesh::addInstancedBuffers

Collects identity quads from the GVolume(GNode) tree into an array, 




Repeating identity guint4 for all volumes of an instance (typically ~5 volumes for 1 instance)
into all the instances (typically large 500-36k).

Instances need to know the sensor they correspond 
even though their geometry is duplicated. 

For analytic geometry this is needed at the volume level 
ie need buffer of size: num_transforms * num_volumes-per-instance

For triangulated geometry this is needed at the triangle level
ie need buffer of size: num_transforms * num_triangles-per-instance

The triangulated version can be created from the analytic one
by duplication according to the number of triangles.





Prior to Aug 2020 this returned an iidentity buffer with all nodes 
when invoked on the root node, eg::  

    GMergedMesh/0/iidentity.npy :       (1, 316326, 4)

This was because of a fundamental difference between the repeated instances and the 
remainder ridx 0 volumes. The volumes of the instances are all together in subtrees 
whereas the remainder volumes with ridx 0 are scattered all over the full tree. Thus 
the former used of this with GGeo::m_root as the only placement resulted in getting 
base + progeny covering all nodes of the tree. To avoid this a separate getRemainderProgeny 
is now used which selects the collected nodes based on the ridx (getRepeatIndex()) 
being zero.

**/

NPY<unsigned int>* GTree::makeInstanceIdentityBuffer(const std::vector<const GNode*>& placements)  // static
{
    unsigned numInstances = placements.size() ;
    const GNode* first_base = placements[0] ;
    GNode* first_base_ = const_cast<GNode*>(first_base); 

    unsigned first_ridx = first_base->getRepeatIndex() ; 
    bool is_remainder = first_ridx == 0 ; 

    if(is_remainder)
    {
        assert( numInstances == 1 );  // only one placement (the root node) for the remainder mm 
    }

   
    std::vector<GNode*>& progeny0 = is_remainder ? first_base_->getRemainderProgeny()           : first_base_->getProgeny();
    unsigned numProgeny0          = is_remainder ? first_base_->getPriorRemainderProgenyCount() : first_base_->getPriorProgenyCount();
    assert( progeny0.size() == numProgeny0 );

    unsigned numVolumes  = 1 + numProgeny0  ;  // "1 +" as progeny does not include base node
    unsigned num = numVolumes*numInstances ; 

    NPY<unsigned>* buf = NPY<unsigned>::make(0, 4);
    NPY<unsigned>* buf2 = NPY<unsigned>::make(numInstances, numVolumes, 4);
    buf2->zero(); 

    for(unsigned int i=0 ; i < numInstances ; i++)
    {
        const GNode* base = placements[i] ; // for global only one placement 
        GNode* base_ = const_cast<GNode*>(base);    // due to progeny cache

        unsigned ridx = base->getRepeatIndex();
        assert( ridx == first_ridx ) ; // all placements by definition have the same ridx 

        std::vector<GNode*>& progeny = is_remainder ? base_->getRemainderProgeny()           : base_->getProgeny() ;
        unsigned numProgeny =          is_remainder ? base_->getPriorRemainderProgenyCount() : base_->getPriorProgenyCount();

        // For the remainder the "progeny" are nodes scattered all over the geometry tree, for the instanced
        // the progeny are nodes in contiguous subtrees.

        assert( numProgeny == numProgeny0 && "repeated geometry for the instances, so the progeny counts must match");

        bool progeny_match = progeny.size() == numProgeny ;

        {
           if(!progeny_match)
           LOG(fatal) 
                      << " progeny_match " << ( progeny_match ? " OK " : " MISMATCH " )
                      << " progeny.size() " << progeny.size() 
                      << " numProgeny " << numProgeny
                      << " numInstances " << numInstances
                      << " numVolumes " << numVolumes
                      << " i " << i 
                      << " ridx " << ridx
                      ;


            assert( progeny_match );
        }

        for(unsigned s=0 ; s < numVolumes ; s++ )
        {
            const GNode* node = s == 0 ? base : progeny[s-1] ; 
            const GVolume* volume = dynamic_cast<const GVolume*>(node) ;

            guint4 id = volume->getIdentity();
            buf->add(id.x, id.y, id.z, id.w ); 

            glm::uvec4 id_ = id.as_vec(); 
            buf2->setQuad( id_, i, s, 0) ; 


#ifdef DEBUG
            std::cout  
                  << " i " << i
                  << " s " << s
                  << " node/mesh/boundary/sensor " << id.x << "/" << id.y << "/" << id.z << "/" << id.w 
                  << " nodeName " << node->getName()
                  << std::endl 
                  ;
#endif
        }
    }
    assert(buf->getNumItems() == num);
    buf->reshape(-1, numVolumes, 4) ; 
    assert(buf->getNumItems() == numInstances);
    assert(buf->hasShape(numInstances,numVolumes, 4));

    bool dump = false ;
    unsigned mismatch = NPY<unsigned>::compare( buf, buf2, dump ); 
    if( mismatch > 0 )
    {
         LOG(fatal) << " buf/buf2 mismatched " << mismatch ; 
         dump = true ; 
         NPY<unsigned>::compare( buf, buf2, dump ); 
    }
    assert( mismatch == 0 );  


    return buf ;  
}

/*

::

    In [1]: ii = np.load("iidentity.npy")

    In [3]: ii.shape
    Out[3]: (3360, 4)

    In [4]: ii.reshape(-1,5,4)
    Out[4]: 
    array([[[ 3199,    47,    19,     1],
            [ 3200,    46,    20,     2],
            [ 3201,    43,    21,     3],
            [ 3202,    44,     1,     4],
            [ 3203,    45,     1,     5]],

           [[ 3205,    47,    19,     6],
            [ 3206,    46,    20,     7],
            [ 3207,    43,    21,     8],
            [ 3208,    44,     1,     9],
            [ 3209,    45,     1,    10]],

After requiring an associated sensor surface to provide the sensor index, only cathodes 
have non-zero index::

    In [1]: ii = np.load("iidentity.npy")

    In [2]: ii.reshape(-1,5,4)
    Out[2]: 
    array([[[ 3199,    47,    19,     0],
            [ 3200,    46,    20,     0],
            [ 3201,    43,    21,     3],
            [ 3202,    44,     1,     0],
            [ 3203,    45,     1,     0]],

           [[ 3205,    47,    19,     0],
            [ 3206,    46,    20,     0],
            [ 3207,    43,    21,     8],
            [ 3208,    44,     1,     0],
            [ 3209,    45,     1,     0]],
*/


