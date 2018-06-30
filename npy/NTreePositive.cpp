#include <iostream>
#include <sstream>

#include "NTreePositive.hpp"
#include "NNodeAnalyse.hpp"
#include "NNodeCollector.hpp"
#include "PLOG.hh"



template <typename T>
int NTreePositive<T>::fVerbosity = 3 ; 
 

template <typename T>
NTreePositive<T>::NTreePositive( T* root )
    :
    m_root(root)
{
    init(); 
} 


template <typename T>
T* NTreePositive<T>::root() const 
{
   return m_root ; 
}


template <typename T>
void NTreePositive<T>::analyse()
{
    //delete m_ana ; 
    m_ana  = new NNodeAnalyse<T>(m_root) ; 

    if(fVerbosity > 2 )
        LOG(info) << " NNodeAnalyse \n" << m_ana->desc(); 
}


template <typename T>
void NTreePositive<T>::init()
{
    if(fVerbosity > 3 )
        LOG(error) << "positivize" ; 

    positivize_r(m_root, false, 0 );

    if(fVerbosity > 3 )
        LOG(error) << "positivize DONE" ; 
}

template <typename T>
void NTreePositive<T>::positivize_r(T* node, bool negate, unsigned depth)
{
    if(fVerbosity > 3 )
        LOG(error) << "positivize_r"
                   << " negate " << negate  
                   << " depth " << depth 
                   ; 

    if(node->left == NULL && node->right == NULL)
    {
        if(negate) node->complement = !node->complement ; 
    } 
    else
    {
        bool left_negate = false ; 
        bool right_negate = false ; 

        if(node->type == CSG_INTERSECTION || node->type == CSG_UNION)
        {
            if(negate)
            {                                   //  !( A*B ) ->  !A + !B       !(A+B) ->     !A * !B
                 node->type = CSG_DeMorganSwap(node->type) ; 
                 left_negate = true ; 
                 right_negate = true ; 
            }
            else
            {                                  //  A * B ->  A * B         A+B ->  A+B
                 left_negate = false ; 
                 right_negate = false ; 
            }
        } 
        else if(node->type == CSG_DIFFERENCE)
        {
            if(negate)                         //  !(A - B) -> !(A*!B) -> !A + B
            {
                node->type = CSG_UNION ;
                left_negate = true ;
                right_negate = false  ;
            }
            else
            {
                node->type = CSG_INTERSECTION ;  //    A - B ->  A*!B
                left_negate = false ;
                right_negate = true ;
            }
        }

        positivize_r(node->left, negate=left_negate, depth=depth+1);
        positivize_r(node->right, negate=right_negate, depth=depth+1);
    }
}





/** 
        deMorganSwap = {CSG_.INTERSECTION:CSG_.UNION, CSG_.UNION:CSG_.INTERSECTION }

        def positivize_r(node, negate=False, depth=0):


            if node.left is None and node.right is None:
                if negate:
                    node.complement = not node.complement
                pass
            else:
                #log.info("beg: %s %s " % (node, "NEGATE" if negate else "") ) 
                if node.typ in [CSG_.INTERSECTION, CSG_.UNION]:

                    if negate:    #  !( A*B ) ->  !A + !B       !(A+B) ->     !A * !B
                        node.typ = deMorganSwap.get(node.typ, None)
                        assert node.typ
                        left_negate = True 
                        right_negate = True
                    else:        #   A * B ->  A * B         A+B ->  A+B
                        left_negate = False
                        right_negate = False
                    pass
                elif node.typ == CSG_.DIFFERENCE:

                    if negate:  #      !(A - B) -> !(A*!B) -> !A + B
                        node.typ = CSG_.UNION 
                        left_negate = True
                        right_negate = False 
                    else:
                        node.typ = CSG_.INTERSECTION    #    A - B ->  A*!B
                        left_negate = False
                        right_negate = True 
                    pass
                else:
                    assert 0, "unexpected node.typ %s " % node.typ
                pass

                #log.info("end: %s " % node ) 
                positivize_r(node.left, negate=left_negate, depth=depth+1)
                positivize_r(node.right, negate=right_negate, depth=depth+1)
            pass
        pass
        positivize_r(self)
**/








#include "NNode.hpp"
#include "No.hpp"
template class NTreePositive<nnode> ; 
template class NTreePositive<no> ; 

