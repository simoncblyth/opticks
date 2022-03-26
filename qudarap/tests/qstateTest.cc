#include "QState.hh"

const char* FOLD = "/tmp/QStateTest" ; 

int main(int argc, char** argv)
{
    qstate s0 = QState::Make(); 
    std::cout << " s0 " << QState::Desc(s0) << std::endl ; 

    QState::Save(s0, FOLD, "s.npy" ); 

    qstate s1 ; 
    QState::Load(s1, FOLD, "s.npy" ); 
    std::cout << " s1 " << QState::Desc(s1) << std::endl ; 

    return 0 ; 
}
