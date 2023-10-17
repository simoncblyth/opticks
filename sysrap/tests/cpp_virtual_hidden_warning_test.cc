// name=cpp_virtual_hidden_warning_test ; gcc $name.cc -std=c++11 -lstdc++ -Wall -Woverloaded-virtual -DWITH_FIX -o /tmp/$name && /tmp/$name

// https://stackoverflow.com/questions/15295317/xcode-why-is-a-warning-of-is-hidden-given-with-overloaded-virtual-functions


#include <cstdio>

class A {
public:
  virtual void methodA(int) { printf("A::methodA(int)\n") ; }
  virtual void methodA(int, int, int) { printf("A::methodA(int, int, int)\n") ; }
};

class B : public A {
public:

#ifdef WITH_FIX
  using A::methodA;  //bring all overloads of methodA into B's scope 
#endif

  virtual void methodA(int) { printf("B::methodA(int)\n") ; }
};

/**
Problem with B is that it overrides only one of the methodA
overloads which causes the other overload to be hidden.  

**/




int main()
{
    A a;
    printf("a.methodA(7)\n"); 
    a.methodA(7);       //OK
    
    printf("a.methodA(7,7,7)\n"); 
    a.methodA(7, 7, 7); //OK

    B b;
    A *pa = &b;

    printf("pa->methodA(7) : calls B impl \n"); 
    pa->methodA(7);        //OK, calls B's implementation

    printf("pa->methodA(7,7,7) : calls A impl \n"); 
    pa->methodA(7, 7, 7);  //OK, calls A's implementation


    b.methodA(7); //OK

#ifdef WITH_FIX
    b.methodA(7, 7, 7);  
#else
    // without the fix : get compile error : methodA of B only accepts one int, not three.
#endif

    return 0 ; 
}
