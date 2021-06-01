#include "OPTICKS_LOG.hh"

struct Base
{
   Base(int base_ ) 
      :
      base(base_)
   {
   }

   virtual void dump(const char* msg) = 0 ; 
   int base ; 

   static void Dump(Base* obj, const char* msg)
   {
       if(obj)
       {
           obj->dump(msg) ; 
       }
       else
       {
           LOG(info) << " NULL " << msg ; 
       }
   }

};

struct A : public Base 
{

   A(int a0_) 
      :
      Base(100),
      a0(a0_)
   {
   }  

   int a0 ;

   void dump(const char* msg)
   {
       LOG(info) 
          << msg 
          << " Base.base " << base 
          << " A.a0 " << a0
          ; 
   }

};

struct B : public Base 
{
   B(int b0_) 
      :
      Base(200),
      b0(b0_)
   {
   }  

   void dump(const char* msg)
   {
       LOG(info) 
          << msg 
          << " Base.base " << base 
          << " B.b0 " << b0
          ; 
   }

   int b0 ;
};



void test_cast()
{
    Base* x = new A(42) ; 


    A* a_dc = dynamic_cast<A*>(x) ;  
    Base::Dump(a_dc, "a_dc");  

    A* a_rc = reinterpret_cast<A*>(x) ;   
    Base::Dump(a_rc, "a_rc");  

    A* a_sc = static_cast<A*>(x) ;   
    Base::Dump(a_sc, "a_sc");  



    B* b_dc = dynamic_cast<B*>(x) ;    // <--- only this one notices are casting to the wrong type and gives NULL
    Base::Dump(b_dc, "b_dc");  

    B* b_rc = reinterpret_cast<B*>(x) ;   
    Base::Dump(b_rc, "b_rc");  

    B* b_sc = static_cast<B*>(x) ;   
    Base::Dump(b_sc, "b_sc");  
}



void test_cast_2()
{
     int a,b,c,d;
     char h = 1;
     short s = 1;
     void* p = nullptr;
     int i = 1;

     assert( sizeof(long) == 8 ); 

     std::cout << "a is at: " << reinterpret_cast<long>(&a) << std::endl;
     std::cout << "b is at: " << reinterpret_cast<long>(&b) << std::endl;
     std::cout << "c is at: " << reinterpret_cast<long>(&c) << std::endl;
     std::cout << "d is at: " << reinterpret_cast<long>(&d) << std::endl;
     std::cout << "Char is at: " << reinterpret_cast<long>(&h) << std::endl;
     std::cout << "Short is at: " << reinterpret_cast<long>(&s) << std::endl;
     std::cout << "Pointer is at: " << reinterpret_cast<long>(p) << std::endl;
     std::cout << "Int is at: " << reinterpret_cast<long>(&i) << std::endl;
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    //test_cast(); 
  
    test_cast_2(); 

    return 0 ;
}

//om- ; TEST=CastTest om-t 


