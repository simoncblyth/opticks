rsync_put_repo_macOS_to_Linux_case_irregularity
===================================================

Issue, on Linux end after::

    ~/opticks/bin/rsync_put.sh  


Case irregularity, seen in the rsynced repo on the Linux end::

    N[blyth@localhost junosw]$ o
    On branch master
    Your branch is up to date with 'origin/master'.

    Changes not staged for commit:
      (use "git add/rm <file>..." to update what will be committed)
      (use "git restore <file>..." to discard changes in working directory)
        deleted:    qudarap/tests/qstateTest.cc

    no changes added to commit (use "git add" and/or "git commit -a")


Use "git show HEAD:..."  to work out what is wrong::

    epsilon:opticks blyth$ git show HEAD:qudarap/tests/qstateTest.cc
    #include "QState.hh"

    const char* FOLD = "/tmp/QStateTest" ; 

    int main(int argc, char** argv)
    {
        sstate s0 = QState::Make(); 
        std::cout << " s0 " << QState::Desc(s0) << std::endl ; 

        QState::Save(s0, FOLD, "s.npy" ); 

        sstate s1 ; 
        QState::Load(s1, FOLD, "s.npy" ); 
        std::cout << " s1 " << QState::Desc(s1) << std::endl ; 

        return 0 ; 
    }
    epsilon:opticks blyth$ 
    epsilon:opticks blyth$ git show HEAD:qudarap/tests/QStateTest.cc
    fatal: Path 'qudarap/tests/QStateTest.cc' exists on disk, but not in 'HEAD'.
    epsilon:opticks blyth$ 

Looks like renamed in file system bit not in git repo, on macOS that 
is not noticed but it is in git. 

Sort it out on laptop::

    epsilon:tests blyth$ mv QStateTest.cc qstateTest.cc
    epsilon:tests blyth$ git mv qstateTest.cc QStateTest.cc

After another sync the irregulatity is cleared up::

    N[blyth@localhost opticks]$ o
    On branch master
    Your branch is up to date with 'origin/master'.

    nothing to commit, working tree clean
    N[blyth@localhost opticks]$ 



