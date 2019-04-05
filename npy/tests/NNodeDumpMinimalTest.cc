// TEST=NNodeDumpMinimalTest om-t 

#include "NSphere.hpp"

#include "NNode.hpp"
#include "NNodeSample.hpp"
#include "OPTICKS_LOG.hh"


/*

   THE PROBLEMS COMMENTED HERE WERE ARE WITH gcc 5.4.0 on Ubuntu 16 
   PRIOR TO MOVING PRIMITIVE CONSTRUCTION TO THE HEAP 

   THIS IS TO AVOID WHAT LOOKS LIKE A GCC 5.4.0 REGRESSION 
   RELATED TO THE IMPLICIT COPY CTOR 

   MOVING TO THE POINTER APPROACH FOR PRIMITIVES REMOVES
   DEPENDENCY ON THE IMPLICIT COPY CTOR THAT SEEMS
   BROKEN IN GCC 5.4.0
    

*/




void t0()   // infinite loops for gcc 5.4.0
{
    LOG(info) ; 
    typedef std::vector<nnode*> VN ;
    VN nodes ; 
    NNodeSample::Tests(nodes);

    nodes[0]->dump(); 
}
void t1()   // fails to dynamic_cast node of type CSG_SPHERE to nsphere
{
    LOG(info) ; 
    nnode* n = NNodeSample::Sphere1() ; 
    n->dump() ;
}



void t1c()  // works 
{
    LOG(info); 
    nsphere* o = make_sphere(0.f,0.f,-50.f,100.f);
    nnode* n = o ; 
    n->dump();
}


/*

void t1d()  // fails : so the problem is related to the original object going out of scope : somehow handled different in gcc 5.4.0
{
    LOG(info); 

    nsphere* a = NULL ; 
    {
        nsphere o = make_sphere(0.f,0.f,-50.f,100.f);
        // why should o going out of scope matter ? 
        // perhaps implicit copy ctor is being overly lazy : overly agressive optimization  ??
        // :google:`gcc 5.4 optimization bug` 

        a = new nsphere(o) ; 
    }
    nnode* n = a ; 
    n->dump();
}


void t1f()   // fails, commenting out the scope braces and it works 
{
    LOG(info); 
    nnode* a = NULL ; 
    {
        nsphere o = make_sphere(0.f,0.f,-50.f,100.f);
        o.dump(); 
        a = new nsphere(o) ;
    }
    a->dump();
}



*/




struct A
{
    const char* label ; 
    int value ; 
    nquad param ; 

    void dump()
    {
        std::cout << "A::dump l:" << label << " v:" << value << " p.f.x:" << param.f.x  << std::endl ; 
    }
};

void t1g() // works : so whats different with nnode and nsphere  
{
    LOG(info); 
    const char* msg = "hi" ; 
    int x = 42 ; 
    A* a = NULL ; 
    {
        A o ; 
        o.label = strdup(msg) ; 
        o.value = x ;
        o.param.f = { 1.f, 2.f, 3.f, 4.f } ; 

        o.dump();   

        a = new A(o) ;  
    } 
    assert( a->value == x );
    assert( strcmp( a->label, msg ) == 0) ; 
    a->dump(); 
}



struct AA : A 
{
    void dump()
    {
        std::cout << "AA::dump l:" << label << " v:" << value << " p.f.x:" << param.f.x  << std::endl ; 
    }
};


void t1h()  // works
{
    LOG(info); 
     
    const char* msg = "hi" ; 
    int x = 42 ; 
    AA* a = NULL ; 
    {
        AA o ; 
        o.label = strdup(msg) ; 
        o.value = x ;
        o.dump();   

        a = new AA(o) ;  
    } 
    assert( a->value == x );
    assert( strcmp( a->label, msg ) == 0) ; 
    a->dump(); 
}





/*

void t2()  // fails to dynamic_cast node of type CSG_SPHERE to nsphere
{
    LOG(info) ; 
    nsphere* a = new nsphere(make_sphere(0.f,0.f,-50.f,100.f));
    a->dump(); 
}

void t3()  // works for gcc 5.4.0
{
    LOG(info) ; 
    nsphere a = make_sphere(0.f,0.f,-50.f,100.f);
    a.dump(); 
}

void t3f()  // working still
{
    LOG(info) ; 

    nsphere a = make_sphere(0.f,0.f,-50.f,100.f);
    a.dump(); 

    nnode* n = (nnode*)&a ; 
    n->dump(); 
}

void t4()  // works too 
{
    nsphere a = make_sphere(0.f,0.f,-50.f,100.f);
    a.dump();

    nsphere b(a) ; 
    b.dump(); 
}

void t5()  //  works too
{
    nsphere a = make_sphere(0.f,0.f,-50.f,100.f);
    a.dump();

    nsphere* b = new nsphere(a) ; 
    b->dump(); 
}

*/





int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    t1();

    return 0 ; 
}



