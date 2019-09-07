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

// this functionality moved into GTree and invoked from GMergedMesh::addInstancedBuffers

/*

void GScene::makeInstancedBuffers(GMergedMesh* mm, unsigned ridx, bool honour_selection)
{
    bool inside = ridx == 0 ; 
    const std::vector<GNode*>& instances = m_root->findAllInstances(ridx, inside, honour_selection );
    unsigned num_instances = instances.size(); 

    if(m_verbosity > 1) 
    LOG(info) << "GScene::makeInstancedBuffers" 
              << " ridx " << std::setw(3) << ridx
              << " num_instances " << std::setw(5) << num_instances
              ;

    NPY<float>* itr = makeInstanceTransformsBuffer(instances, ridx); 
    mm->setITransformsBuffer(itr);

    NPY<unsigned>* iid = makeInstanceIdentityBuffer(instances, ridx);
    mm->setInstancedIdentityBuffer(iid);

    NPY<unsigned>* aii = makeAnalyticInstanceIdentityBuffer(instances, ridx);
    mm->setAnalyticInstancedIdentityBuffer(aii);
}


NPY<float>* GScene::makeInstanceTransformsBuffer(const std::vector<GNode*>& instances, unsigned ridx)
{
    NPY<float>* buf = NULL ; 
    if(ridx == 0)
    {
        buf = NPY<float>::make_identity_transforms(1) ; 
    }
    else
    {
        unsigned num_instances = instances.size(); 
        buf = NPY<float>::make(num_instances, 4, 4);
        buf->zero(); 
        for(unsigned i=0 ; i < num_instances ; i++)
        {
            GNode* instance = instances[i] ;
            GMatrix<float>* gtransform = instance->getTransform();
            const float* data = static_cast<float*>(gtransform->getPointer());
            glm::mat4 xf_global = glm::make_mat4( data ) ;  
            buf->setMat4(xf_global, i);  
        } 
    }
    return buf ; 
}

NPY<unsigned>* GScene::makeInstanceIdentityBuffer(const std::vector<GNode*>& instances, unsigned ridx)   
{
    unsigned num_instances = instances.size(); 
    NPY<unsigned>* buf = NPY<unsigned>::make(num_instances, 4);
    buf->zero(); 

    // huh : this is not doing the solid duplication of GInstancer::makeInstanceIdentityBuffer  ???


    for(unsigned i=0 ; i < num_instances ; i++)
    {
        GSolid* instance = dynamic_cast<GSolid*>(instances[i]) ; 
        guint4 id = instance->getIdentity();
        glm::uvec4 uid(id.x, id.y, id.z, id.w);
        buf->setQuadU( uid, i );
    }
    return buf ;  
}

NPY<unsigned>* GScene::makeAnalyticInstanceIdentityBuffer( const std::vector<GNode*>& instances, unsigned ridx) 
{
    unsigned num_instances = instances.size(); 
    NPY<unsigned>* buf = NPY<unsigned>::make(num_instances, 1, 4);  //  TODO: unify shape aii and ii shape
    buf->zero(); 

    for(unsigned int i=0 ; i < num_instances ; i++) // over instances of the same geometry
    {
        GSolid* instance = dynamic_cast<GSolid*>(instances[i]) ; 
        NSensor* ss = instance->getSensor();
        unsigned int sid = ss && ss->isCathode() ? ss->getId() : 0 ;

        if(sid > 0)
            LOG(debug) << "GScene::makeAnalyticInstanceIdentityBuffer " 
                      << " sid " << std::setw(10) << std::hex << sid << std::dec 
                      << " ss " << (ss ? ss->description() : "NULL" )
                      ;

        glm::uvec4 aii ; 
        aii.x = instance->getIndex()  ;        
        aii.y = i ;                      // instance index (for triangulated this contains the mesh index)
        aii.z = 0 ;                      // formerly boundary, but with analytic have broken 1-1 solid/boundary relationship so boundary must live in partBuffer
        aii.w = NSensor::RefIndex(ss) ;  // the only critical one 

        buf->setQuadU(aii, i, 0); 
        
    }
    return buf ; 
}

*/



