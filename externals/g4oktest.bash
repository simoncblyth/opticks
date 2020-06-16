g4oktest-source(){   echo $BASH_SOURCE ; }
g4oktest-vi(){       vi $BASH_SOURCE ; }
g4oktest-env(){      olocal- ; }
g4oktest-usage(){ cat << \EOU

G4OpticksTest : Fork of Hans Repo
===================================

git remote add upstream, git fetch upstream
---------------------------------------------

* https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork

::

    epsilon:G4OpticksTest blyth$ git remote -v
    origin	git@github.com:simoncblyth/G4OpticksTest.git (fetch)
    origin	git@github.com:simoncblyth/G4OpticksTest.git (push)

    epsilon:G4OpticksTest blyth$ git remote add upstream https://github.com/hanswenzel/G4OpticksTest.git
    epsilon:G4OpticksTest blyth$ git remote -v
    origin	git@github.com:simoncblyth/G4OpticksTest.git (fetch)
    origin	git@github.com:simoncblyth/G4OpticksTest.git (push)
    upstream	https://github.com/hanswenzel/G4OpticksTest.git (fetch)
    upstream	https://github.com/hanswenzel/G4OpticksTest.git (push)

    epsilon:G4OpticksTest blyth$ git fetch upstream
    remote: Enumerating objects: 34, done.
    remote: Counting objects: 100% (34/34), done.
    remote: Compressing objects: 100% (14/14), done.
    remote: Total 34 (delta 21), reused 31 (delta 20), pack-reused 0
    Unpacking objects: 100% (34/34), done.
    From https://github.com/hanswenzel/G4OpticksTest
     * [new branch]      master     -> upstream/master
     * [new tag]         v0.1.1     -> v0.1.1
    epsilon:G4OpticksTest blyth$ 


git merge upstream/master
--------------------------

::

    epsilon:G4OpticksTest blyth$ git checkout master
    Already on 'master'
    Your branch is up-to-date with 'origin/master'.


    epsilon:G4OpticksTest blyth$ git merge upstream/master
    Updating b55775f..c3f4aa3
    Fast-forward
     README.md                       |    4 +-
     gdml/CerenkovMinimal.gdml       |   32 +++-
     gdml/G4Opticks.gdml             |  188 +++++++++++++++++++++++
     include/DetConOrg.hh            |   59 ++++++++
     include/DetectorConstruction.hh |    2 +-
     include/G4.hh                   |   53 ++++---
     include/PhysicsList.hh          |    9 +-
     muon.mac                        |    9 ++
     setup_opticks.sh                |   15 +-
     src/DetConOrg.cc                |  291 +++++++++++++++++++++++++++++++++++
     src/DetectorConstruction.cc     |    5 +-
     src/G4.cc                       |   11 +-
     src/L4Scintillation.cc          | 1305 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-------------------------------------------------------------------------------
     src/PhysicsList.cc              |   43 ++++--
     14 files changed, 1308 insertions(+), 718 deletions(-)
     create mode 100644 gdml/G4Opticks.gdml
     create mode 100644 include/DetConOrg.hh
     create mode 100644 muon.mac
     create mode 100644 src/DetConOrg.cc
    epsilon:G4OpticksTest blyth$ 


EOU
}


g4oktest-dir(){ echo $HOME/G4OpticksTest ; } 
g4oktest-get()
{
    cd $(dirname $(g4oktest-dir))
    [ ! -d "G4OpticksTest" ] && git clone git@github.com:simoncblyth/G4OpticksTest.git
}
g4oktest-cd(){ cd $(g4oktest-dir) ; } 
g4oktest-c(){  cd $(g4oktest-dir) ; } 

g4oktest-om()
{
    g4oktest-cd
    ./om.sh 
}

g4oktest--(){
    g4oktest-get
    g4oktest-om
}





