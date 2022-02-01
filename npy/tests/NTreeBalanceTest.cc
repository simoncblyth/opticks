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

// TEST=NTreeBalanceTest om-t

#include <cstdlib>
#include <cassert>
#include <vector>

#include "OPTICKS_LOG.hh"

#include "NBox.hpp"
#include "NSphere.hpp"
#include "NNode.hpp"
#include "NTreeAnalyse.hpp"
#include "NTreePositive.hpp"
#include "NTreeBalance.hpp"
#include "NTreeProcess.hpp"


nnode* make_tree_0()
{
    nnode* a = make_sphere(0,0,-50,100) ;    a->label = "a" ;
    nnode* b = make_sphere(0,0, 50,100) ;    b->label = "b" ;
    nnode* c = make_box(0,0, 50,100) ;       c->label = "c" ; 
    nnode* d = make_box(0,0,  0,100) ;       d->label = "d" ;
    nnode* e = make_box(0,0,  0,100) ;       e->label = "e" ;

    nnode* ab = nnode::make_operator( CSG_UNION, a, b );
    nnode* de = nnode::make_operator( CSG_DIFFERENCE, d, e );
    nnode* cde = nnode::make_operator( CSG_DIFFERENCE, c, de );
    nnode* abcde = nnode::make_operator( CSG_INTERSECTION, ab, cde );

    return abcde ; 
}


nnode* make_tree_1(OpticksCSG_t op)
{
    nnode* a = make_sphere(0,0,-50,100) ;  a->label = "a" ; 
    nnode* b = make_sphere(0,0,-50,100) ;  b->label = "b" ;  
    nnode* c = make_sphere(0,0,-50,100) ;  c->label = "c" ; 
    nnode* d = make_sphere(0,0,-50,100) ;  d->label = "d" ; 
    nnode* e = make_sphere(0,0,-50,100) ;  e->label = "e" ; 
    nnode* f = make_sphere(0,0,-50,100) ;  f->label = "f" ;  
    nnode* g = make_sphere(0,0,-50,100) ;  g->label = "g" ; 

    nnode* ab = nnode::make_operator( op , a, b );  ab->label="ab" ; 
    nnode* abc = nnode::make_operator( op, ab, c ); abc->label="abc" ; 
    nnode* abcd = nnode::make_operator( op, abc, d ); abcd->label="abcd" ; 
    nnode* abcde = nnode::make_operator( op, abcd, e ); abcde->label="abcde" ; 
    nnode* abcdef = nnode::make_operator( op, abcde, f ); abcdef->label="abcdef" ; 
    nnode* abcdefg = nnode::make_operator( op, abcdef, g ); abcdefg->label="abcdefg" ; 

    return abcdefg ; 
}

/**
make_tree_2
--------------

                                                                                      un            

                                                                              un              di    

                                                                      un          cy      cy      cy

                                                              un          cy                        

                                                      un          cy                                

                                              un          cy                                        

                                      un          cy                                                

                              un          cy                                                        

                      un          cy                                                                

              un          cy                                                                        

      di          cy                                                                                

  cy      cy                                    


**/


nnode* make_tree_2()
{
    nnode* a = make_sphere(0,0,-50,100) ;   a->label = "a" ; 
    nnode* b = make_sphere(0,0,-50,100) ;   b->label = "b" ; 
    nnode* c = make_sphere(0,0,-50,100) ;   c->label = "c" ;  
    nnode* d = make_sphere(0,0,-50,100) ;   d->label = "d" ;  
    nnode* e = make_sphere(0,0,-50,100) ;   e->label = "e" ;  
    nnode* f = make_sphere(0,0,-50,100) ;   f->label = "f" ;  
    nnode* g = make_sphere(0,0,-50,100) ;   g->label = "g" ;  
    nnode* h = make_sphere(0,0,-50,100) ;   h->label = "h" ; 
    nnode* i = make_sphere(0,0,-50,100) ;   i->label = "i" ; 
    nnode* j = make_sphere(0,0,-50,100) ;   j->label = "j" ; 
    nnode* k = make_sphere(0,0,-50,100) ;   k->label = "k" ; 
    nnode* l = make_sphere(0,0,-50,100) ;   l->label = "l" ; 
    nnode* m = make_sphere(0,0,-50,100) ;   m->label = "m" ; 

    nnode* ab = nnode::make_operator( CSG_DIFFERENCE, a, b );

    nnode* abc = nnode::make_operator( CSG_UNION, ab, c );
    nnode* abcd = nnode::make_operator( CSG_UNION, abc, d );
    nnode* abcde = nnode::make_operator( CSG_UNION, abcd, e );
    nnode* abcdef = nnode::make_operator( CSG_UNION, abcde, f );
    nnode* abcdefg = nnode::make_operator( CSG_UNION, abcdef, g );
    nnode* abcdefgh = nnode::make_operator( CSG_UNION, abcdefg, h );
    nnode* abcdefghi = nnode::make_operator( CSG_UNION, abcdefgh, i );
    nnode* abcdefghij = nnode::make_operator( CSG_UNION, abcdefghi, j );
    nnode* abcdefghijk = nnode::make_operator( CSG_UNION, abcdefghij, k );

    nnode* lm = nnode::make_operator( CSG_DIFFERENCE, l, m );

    nnode* abcdefghijklm = nnode::make_operator( CSG_UNION, abcdefghijk, lm );
    nnode* root = abcdefghijklm  ; 

    return root ; 
}



nnode* make_tree_3()
{
    nnode* a = make_sphere(0,0,-50,100) ;   a->label = "a" ; 
    nnode* b = make_sphere(0,0,-50,100) ;   b->label = "b" ; 
    nnode* c = make_sphere(0,0,-50,100) ;   c->label = "c" ;  
    nnode* d = make_sphere(0,0,-50,100) ;   d->label = "d" ;  
    nnode* e = make_sphere(0,0,-50,100) ;   e->label = "e" ;  
    nnode* f = make_sphere(0,0,-50,100) ;   f->label = "f" ;  
    nnode* g = make_sphere(0,0,-50,100) ;   g->label = "g" ;  

    nnode* ab = nnode::make_operator( CSG_UNION, a, b );
    nnode* abc = nnode::make_operator( CSG_UNION, ab, c );
    nnode* abcd = nnode::make_operator( CSG_UNION, abc, d );
    nnode* abcde = nnode::make_operator( CSG_UNION, abcd, e );
    nnode* abcdef = nnode::make_operator( CSG_UNION, abcde, f );
    nnode* abcdefg = nnode::make_operator( CSG_DIFFERENCE, abcdef, g );

    return abcdefg  ; 
}



nnode* make_tree_mono(OpticksCSG_t op, unsigned num_leaf)
{
    std::string labels = "abcdefghijklmnopqrstuvwxyz" ; 
    assert( num_leaf < strlen(labels.c_str()) ); 

    nnode* comp = nullptr ; 
    for(unsigned i=0 ; i < num_leaf ; i++) 
    {
         nnode* node = make_sphere(0,0,-50,100) ;   
         node->label = strdup(labels.substr(i,1).c_str()) ; 
         comp = comp == nullptr ? node :  nnode::make_operator( CSG_UNION, comp, node ) ; 
    }
    return comp  ; 
}



nnode* make_tree( int i )
{
    nnode* root = NULL ; 
    switch(i)
    {
        case 0: root = make_tree_0()                  ; break ; 
        case 1: root = make_tree_1(CSG_UNION)         ; break ; 
        case 2: root = make_tree_2()                  ; break ; 
        case 3: root = make_tree_3()                  ; break ; 
        case 4: root = make_tree_mono(CSG_UNION, 4)   ; break ; 
        case 5: root = make_tree_mono(CSG_UNION, 5)   ; break ; 
        case 6: root = make_tree_mono(CSG_UNION, 6)   ; break ; 
    }
    assert( root ) ; 
    return root ; 
}




void test_balance_0(nnode* tree)
{
    bool dump = true ; 
    NTreeBalance<nnode> bal(tree, dump);  // writes depth, subdepth to all nodes

    LOG(info) << tree->desc() ; 
    LOG(info) << NTreeAnalyse<nnode>::Desc(tree) ;
    LOG(info) << " ops: " << bal.operatorsDesc(0) ; 
    assert( bal.is_positive_form() == false );


    NTreePositive<nnode> pos(tree) ; 

    LOG(info) << NTreeAnalyse<nnode>::Desc(tree) ; 
    LOG(info) << " ops: " << bal.operatorsDesc(0) ; 
    assert( bal.is_positive_form() == true );


    std::vector<nnode*> subs ; 
    std::vector<nnode*> otherprim ; 
    unsigned subdepth = 0 ; 
    bal.subtrees( subs, subdepth, otherprim );  // 
    LOG(info) 
        << " subdepth " << subdepth 
        << " subs " << subs.size() 
        ; 

    for(unsigned i=0 ; i < subs.size() ; i++)
    {
        nnode* sub = subs[i] ;
        LOG(info) << NTreeAnalyse<nnode>::Desc(sub) ;
    }
}


void test_balance(nnode* tree)
{
    LOG(info) << "tree initial \n" << NTreeAnalyse<nnode>::Desc(tree) ; 

    NTreePositive<nnode> pos(tree) ; // eliminate CSG_DIFFERENCE by DeMorgan rules, some leaves may be complemented 
    LOG(info) << "tree positivized\n" << NTreeAnalyse<nnode>::Desc(tree) ; 

    bool dump = true ; 
    NTreeBalance<nnode> bal(tree, dump);   // writes depth, subdepth to all nodes

    nnode* balanced = bal.create_balanced(); 

    LOG(info) << "tree balanced\n" << NTreeAnalyse<nnode>::Desc(balanced) ; 
}

void test_process(nnode* tree, int idx)
{
    LOG(info) << "tree " << idx << " initial \n" << NTreeAnalyse<nnode>::Desc(tree) ; 

    bool dump = true ; 
    NTreeProcess<nnode> proc(tree, dump); 

    nnode* result = proc.result ; 
    LOG(info) << "tree " << idx << " result \n" << NTreeAnalyse<nnode>::Desc(result) ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    for( int i=0 ; i < 7 ; i++)
    {
        nnode* tree = make_tree(i) ; 
        test_process(tree, i );
    }

    return 0 ; 
}


