#pragma once

#include <iostream>
#include "NPY_API_EXPORT.hh"


template <class T, int NUM_CHILDREN>
class NPY_API NTraverser {
   public:
       NTraverser(T* root, const char* msg="NTraverser", int verbosity=0, int nodelimit=30) 
           : 
           m_root(root), 
           m_nodecount(0),
           m_nodelimit(nodelimit),
           m_verbosity(verbosity)  
       {
           traverse_r(m_root, 0);
           std::cout 
                << "Traverser "
                << msg 
                << " nodecount " << m_nodecount 
                << std::endl ; 
       }
   private:
       void traverse_r(T* node, int depth)
       {
           m_nodecount++ ; 

           if(m_verbosity > 0 && m_nodecount < m_nodelimit)
           {
               std::cout << "traverse_r" 
                         << " depth " << depth 
                         << " nodecount " << m_nodecount
                         << " nodelimit " << m_nodelimit 
                         << std::endl ; 
           } 

           for(int i=0 ; i < NUM_CHILDREN ; i++)
           {
               T* child = node->children[i] ;
               if(child) traverse_r(child, depth+1 );
           }
       }
   private:
       T*          m_root ; 
       int         m_nodecount ; 
       int         m_nodelimit ; 
       int         m_verbosity ; 
};



template <class T, int NUM_CHILDREN>
class NPY_API NComparer {
       enum {
          NSTATE=4 
       };
   public:
      NComparer(T* a, T* b)
          :
          m_a(a),
          m_b(b),
          m_same(false),
          m_equals_count(0),
          m_content_count(0)
       {
           for(int i=0 ; i < NSTATE ; i++) m_state[i] = 0 ; 
           m_same = equals(m_a, m_b);
       }

      void dump(const char* msg)
      {
          std::cout << msg << " " << ( m_same ? "ARE-SAME" : "NOT-SAME" ) ; 
          int state_tot = 0 ; 
          for(int i=0 ; i < NSTATE ; i++)  
          {
             std::cout << " state " <<  i << " : " << m_state[i] << std::endl ; 
             state_tot += m_state[i] ; 
          }
          std::cout 
                  << " equals_count " << m_equals_count 
                  << " content_count " << m_content_count 
                  << " state_tot " << state_tot 
                  << std::endl ;   
      }  

       bool equal_content( T* a, T* b)
       { 
            m_content_count++ ;  
            return *a == *b ; 
       }

       bool equals( T* a, T* b)
       {
           int ab = ((!!a) << 1 ) | (!!b) ;   //  !!(NULL) -> 0, !!(non-NULL) -> 1
           m_state[ab]++ ;   
           m_equals_count++ ; 

           if( ab < 3)
           {
               return ab == 0 ;  // 0:both NULL, 1/2 one NULL other non-NULL
           } 
           else
           {
               int num_equal = 0 ; 
               for(int i=0 ; i < NUM_CHILDREN ; i++) num_equal += equals( a->children[i], b->children[i] ) ;
               return num_equal == NUM_CHILDREN && equal_content(a, b) ; 
           }
       }

   private:
       T*          m_a ; 
       T*          m_b ;
       bool        m_same ; 
       int         m_state[NSTATE] ; 
       int         m_equals_count ; 
       int         m_content_count ; 

}; 


 

