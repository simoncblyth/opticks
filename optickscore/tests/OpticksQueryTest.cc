
#include <sstream>
#include <iostream>
#include <iomanip>
#include <vector>

#include "OpticksQuery.hh"

#include "OKCORE_LOG.hh"
#include "PLOG.hh"


struct Node 
{
    Node(unsigned idx, Node* parent) 
        : 
         idx(idx), 
         parent(parent),
         left(NULL),
         right(NULL)
         {} ;
 
    unsigned idx ; 
    Node* parent ; 
    Node* left  ; 
    Node* right  ; 
};


struct Tree
{
    unsigned maxdepth ; 
    unsigned count ; 
    Node*    root ; 
    std::vector<Node*> selection ; 

    Tree(unsigned maxdepth) 
        : 
        maxdepth(maxdepth),
        count(0),
        root(make_r(1, NULL, 0))  
    {
    }

    Node* make_r(unsigned idx, Node* parent, unsigned depth)
    {
        Node* node = new Node(idx, parent);
        count++ ; 
        if(depth < maxdepth)
        {
            node->left  = make_r( idx*2 , node, depth+1 );
            node->right = make_r( idx*2+1 , node, depth+1  );
        }
        return node ; 
    }

    Node* first() const { return selection.size() > 0 ? selection[0]  : NULL ; }
    Node* last() const {  return selection.size() > 0 ? selection[selection.size()-1]  : NULL ; }


    std::string desc() const 
    {
        Node* f = first();
        Node* l = last();

        std::stringstream ss ; 
        ss << "Tree " 
           << " maxdepth " << maxdepth 
           << " count " << count
           << " selection " << selection.size()
           << " first " << ( f ? f->idx : 0 )
           << " last " << ( l ? l->idx : 0 )
           ;
        return ss.str();
    }

    void select(OpticksQuery* q)
    {
        selection.clear();
        bool recursive_select = false ; 
        select_r(root, q, 0, recursive_select );
    }

    void select_r(Node* node, OpticksQuery* q, unsigned depth, bool recursive_select)
    {
        bool is_selected = q->selected("dummy", node->idx, depth, recursive_select );

        std::cout << "select_r"
                  << " node.idx " << std::setw(6) << node->idx
                  << " depth " << std::setw(2) << depth
                  << " recursive_select " << ( recursive_select ? "Y" : "N" )
                  << " is_selected " << ( is_selected ? "Y" : "N" )
                  << std::endl 
                  ; 

        if(is_selected) 
        {
          // note not a level order traverse : so will not be in order
           selection.push_back(node);
        }

        if(node->left && node->right)
        {
            select_r(node->left , q, depth+1, recursive_select);
            select_r(node->right, q, depth+1, recursive_select);
        }
    }
};





void test_range()
{
     OpticksQuery q("range:10:20");
     q.dump(); 

     Tree t(4) ; 
     LOG(info) << "initial " << t.desc() ; 

     t.select(&q) ; 
     LOG(info) << "after select " << t.desc() ; 
}


void test_all()
{
     OpticksQuery q("all");
     q.dump(); 

     Tree t(4) ; 
     LOG(info) << "initial " << t.desc() ; 

     t.select(&q) ; 
     LOG(info) << "after select " << t.desc() ; 
}


void test_index()
{
     OpticksQuery q("index:5,depth:10");
     q.dump(); 

     Tree t(4) ; 
     LOG(info) << "initial " << t.desc() ; 

     t.select(&q) ; 
     LOG(info) << "after select " << t.desc() ; 
}



int main(int argc, char** argv)
{
     PLOG_(argc, argv);
     OKCORE_LOG__ ; 

     //test_range();
     test_all();
     //test_index();

     return 0 ; 
}
