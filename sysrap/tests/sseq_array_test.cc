#include "ssys.h"
#include "sseq_array.h"

struct sseq_array_test
{
    static int find_selection_indices_ending_0(); 
    static int find_selection_indices_ending_1(); 
    static int main();
};

inline int sseq_array_test::find_selection_indices_ending_0()
{
   typedef unsigned long long ULL ; 

   ULL fsq_0 = OpticksPhoton::AbbrevToFlagSequence("TO BT BR BT SA") ;
   ULL fsq_1 = OpticksPhoton::AbbrevToFlagSequence("TO BT BR BT SD") ;

   std::vector<sseq> v_all(5);
   v_all[0] = {{ fsq_0, 0 }, {0, 0}} ;
   v_all[1] = {{ fsq_1, 0 }, {0, 0}} ;
   v_all[2] = {{ fsq_0, 0 }, {0, 0}} ;
   v_all[3] = {{ fsq_1, 0 }, {0, 0}} ;
   v_all[4] = {{ fsq_0, 0 }, {0, 0}} ;

   NP* all = NPX::ArrayFromVec<ULL, sseq>(v_all);

   sseq_array sqa(all);

   NP* sel = sqa.create_selection("*SD");

   assert( sqa.selection_indices.size() == 2 );
   assert( sqa.selection_indices[0] == 1 );
   assert( sqa.selection_indices[1] == 3 );

   std::cout << " sel " << ( sel ? sel->sstr() : "-" ) << "\n" ; 

   return 0 ; 
}

inline int sseq_array_test::find_selection_indices_ending_1()
{
   typedef unsigned long long ULL ; 
  
   const char* HISTORY = R"LITERAL(
TO BT BR BT SA
TO BT BR BT SD
TO BT BR BT SA
TO BT BR BT SD
TO BT BR BT SA
)LITERAL"; 

   sseq_array sqa(HISTORY);
   NP* sel = sqa.create_selection("*SD");

   assert( sqa.selection_indices.size() == 2 );
   assert( sqa.selection_indices[0] == 1 );
   assert( sqa.selection_indices[1] == 3 );

   std::cout << " sel " << ( sel ? sel->sstr() : "-" ) << "\n" ; 

   return 0 ; 
}

inline int sseq_array_test::main()
{
    const char* TEST = ssys::getenvvar("TEST", "ALL");
    bool ALL = strcmp(TEST, "ALL") == 0 ;
    int rc = 0 ;
    if(ALL||0==strcmp(TEST,"find_selection_indices_ending_0")) rc += find_selection_indices_ending_0();
    if(ALL||0==strcmp(TEST,"find_selection_indices_ending_1")) rc += find_selection_indices_ending_1();
    return rc ;
}


int main()
{
    return sseq_array_test::main();
}


