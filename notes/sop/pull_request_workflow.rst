pull_request_workflow
=======================





manual approach to check PR
----------------------------

::

    # Fetch the PR branch (replace <PR_NUMBER> with the actual PR number)
    git fetch origin pull/<PR_NUMBER>/head:pr-branch

    # Switch to that branch
    git checkout pr-branch




gh cli tool setup
-------------------

0. make sure GH_TOKEN is defined::

   echo $GH_TOKEN

1. check its valid::

   gh auth status




Use gh to checkout the PR branch
--------------------------------


1. high level fetch and checkout::

   cd ~/opticks

   gh pr list
   gh pr checkout 14   # using number of the PR to check from the list


* fetches the branch from the contributors fork
* switches to that branch locally - and sets up upstream tracking


::

    A[blyth@localhost opticks]$ gh pr checkout 14
    Ncat: Version 7.92 ( https://nmap.org/ncat )
    Ncat: Connected to proxy 127.0.0.1:8080
    Ncat: No authentication needed.
    Ncat: Host github.com will be resolved by the proxy.
    Ncat: connection succeeded.
    remote: Enumerating objects: 36, done.
    remote: Counting objects: 100% (33/33), done.
    remote: Compressing objects: 100% (3/3), done.
    remote: Total 18 (delta 15), reused 18 (delta 15), pack-reused 0 (from 0)
    Unpacking objects: 100% (18/18), 3.73 KiB | 382.00 KiB/s, done.
    From github.com:simoncblyth/opticks
     * [new ref]             refs/pull/14/head -> yangdy_three_species_LS_model_pr1_sspecies_sphoton_genstep
    Switched to branch 'yangdy_three_species_LS_model_pr1_sspecies_sphoton_genstep'
    A[blyth@localhost opticks]$
    A[blyth@localhost opticks]$
    A[blyth@localhost opticks]$ git status
    On branch yangdy_three_species_LS_model_pr1_sspecies_sphoton_genstep
    nothing to commit, working tree clean
    A[blyth@localhost opticks]$


Merge master branch into PR branch::

    git merge master




If there are conflicts - do fixing pushes go into the contributors fork ? Surely I would not have permissions for that ? YOU DO AND IT DOES GO TO THEIR FORK
-------------------------------------------------------------------------------------------------------------------------------------------------------------

* When a contributor opens a pull request from a fork on GitHub,
  there is a checkbox on their end called "Allow edits from maintainers" (which is checked by default).

When you run git push after using gh pr checkout <PR_NUMBER>, it automatically configures your local branch tracking
remote to point directly to the contributor fork and their specific branch.

* Your git push goes straight into their fork, which instantly updates the open pull request on your repository with your conflict resolution.



Checking the merged PR
------------------------


0. setup build env : lo
1. check it builds : oo
2. run tests : opticks-t



Proceed to merge
-------------------


Once pushed, the PR on GitHub will automatically update to show that conflicts
are resolved. You can either:

1.  merge it directly through the GitHub web UI
2.  OR use the CLI::

    gh pr merge 14 --merge --delete-branch


* after merging using the GitHub web UI - leave a Thank you comment
* if you want to decline the contribution, use : "Close and Comment"



Return to master and update
---------------------------

::

    git checkout master
    git pull origin master


Cleanup the now merged branch
------------------------------

::

    [lo] A[blyth@localhost opticks]$ git branch
    * master
      yangdy_three_species_LS_model_pr1_sspecies_sphoton_genstep
    [lo] A[blyth@localhost opticks]$ git branch -d yangdy_three_species_LS_model_pr1_sspecies_sphoton_genstep
    error: the branch 'yangdy_three_species_LS_model_pr1_sspecies_sphoton_genstep' is not fully merged
    hint: If you are sure you want to delete it, run 'git branch -D yangdy_three_species_LS_model_pr1_sspecies_sphoton_genstep'
    hint: Disable this message with "git config advice.forceDeleteBranch false"
    [lo] A[blyth@localhost opticks]$ git branch -D yangdy_three_species_LS_model_pr1_sspecies_sphoton_genstep
    Deleted branch yangdy_three_species_LS_model_pr1_sspecies_sphoton_genstep (was 6d682050c).
    [lo] A[blyth@localhost opticks]$ git branch
    * master
    [lo] A[blyth@localhost opticks]$


List hashes of the merged commits
-----------------------------------------------

::

    [lo] A[blyth@localhost opticks]$ git log --author=yangdy --oneline
    a712886ff Merge branch 'master' into yangdy_three_species_LS_model_pr1_sspecies_sphoton_genstep
    ce1050f03 add species tag to sphoton and DsG4ScintSimple genstep type for JUNO three-species LS model.



Check again the build and tests
---------------------------------


::

    oo







