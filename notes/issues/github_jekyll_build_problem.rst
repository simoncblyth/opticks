github_jekyll_build_problem : FIXED BY DELETING SYMBOLIC LINKS IN DIR /env/Documents/Geant4OpticksWorkflow 
=============================================================================================================

::

    cd ~/simoncblyth.bitbucket.io/
    git push github


After push to github, get below email, but the pages appear OK::

    The page build failed for the `master` branch with the following error:

    Site contained a symlink that should be dereferenced: /env/Documents/Geant4OpticksWorkflow/Geant4OpticksWorkflow1_001.png. 

    For more information, 

    see https://docs.github.com/github/working-with-github-pages/troubleshooting-jekyll-build-errors-for-github-pages-sites#config-file-error.

    For information on troubleshooting Jekyll see:

     https://docs.github.com/articles/troubleshooting-jekyll-builds

    If you have any questions you can submit a request at https://support.github.com/contact?page_build_id=438482787&repo_id=553516581&tags=dotcom-pages



In happens again::

    The page build failed for the `master` branch with the following error:

    Site contained a symlink that should be dereferenced: /env/Documents/Geant4OpticksWorkflow/Geant4OpticksWorkflow1_001.png. For more information, see https://docs.github.com/github/working-with-github-pages/troubleshooting-jekyll-build-errors-for-github-pages-sites#config-file-error.

    For information on troubleshooting Jekyll see:

     https://docs.github.com/articles/troubleshooting-jekyll-builds

    If you have any questions you can submit a request at https://support.github.com/contact?page_build_id=441702859&repo_id=553516581&tags=dotcom-pages


Note that this time the error seems to prevent any updates on the server. 


::

    epsilon:issues blyth$ cd /env/Documents/Geant4OpticksWorkflow
    epsilon:Geant4OpticksWorkflow blyth$ l
    total 1048
     80 -rw-r--r--@ 1 blyth  staff   39917 Jan 26  2022 Geant4OpticksWorkflow.001_thumb4.png
      0 drwxr-xr-x  7 blyth  staff     224 Jan 26  2022 .
      0 lrwxr-xr-x  1 blyth  staff      30 May 17  2020 Geant4OpticksWorkflow1_001.png -> Geant4OpticksWorkflow1.001.png
      0 lrwxr-xr-x  1 blyth  staff      29 May 17  2020 Geant4OpticksWorkflow_001.png -> Geant4OpticksWorkflow.001.png
    472 -rw-r--r--@ 1 blyth  staff  241294 May 17  2020 Geant4OpticksWorkflow1.001.png
      0 drwxr-xr-x  4 blyth  staff     128 May 16  2020 ..
    496 -rw-r--r--@ 1 blyth  staff  251585 May 16  2020 Geant4OpticksWorkflow.001.png
    epsilon:Geant4OpticksWorkflow blyth$ 


It looks like some tool had issues with the double dot which was kludge fixed with the symbolic link. 



* https://github.com/orgs/community/discussions/9104



Curious, there is no html or source with "Geant4OpticksWorkflow1" so issue not with referencing but rather with the actual png and symlink ?::

    epsilon:simoncblyth.bitbucket.io blyth$ find . -name '*.html'  -exec grep -H Geant4OpticksWorkflow1 {} \;
    epsilon:simoncblyth.bitbucket.io blyth$ 

    epsilon:simoncblyth.bitbucket.io blyth$ presentation-cd
    epsilon:presentation blyth$ pwd
    /Users/blyth/env/presentation
    epsilon:presentation blyth$ find . -type f -exec grep -H Geant4OpticksWorkflow1 {} \;
    epsilon:presentation blyth$ 


Looks like no html is using the underscore png::


    epsilon:simoncblyth.bitbucket.io blyth$ find . -name '*.html' -exec grep -H Geant4OpticksWorkflow_ {} \;
    epsilon:simoncblyth.bitbucket.io blyth$ 
    epsilon:simoncblyth.bitbucket.io blyth$ find . -name '*.html' -exec grep -H Geant4OpticksWorkflow1_ {} \;
    epsilon:simoncblyth.bitbucket.io blyth$ 


So, try simply removing the symlinks::

    epsilon:Geant4OpticksWorkflow blyth$ cd ~/simoncblyth.bitbucket.io/env/Documents/Geant4OpticksWorkflow
    epsilon:Geant4OpticksWorkflow blyth$ l
    total 1048
      0 drwxr-xr-x  5 blyth  staff     160 Apr 26 16:34 ..
     80 -rw-r--r--@ 1 blyth  staff   39917 Jan 26  2022 Geant4OpticksWorkflow.001_thumb4.png
      0 drwxr-xr-x  7 blyth  staff     224 Jan 26  2022 .
      0 lrwxr-xr-x  1 blyth  staff      30 May 17  2020 Geant4OpticksWorkflow1_001.png -> Geant4OpticksWorkflow1.001.png
      0 lrwxr-xr-x  1 blyth  staff      29 May 17  2020 Geant4OpticksWorkflow_001.png -> Geant4OpticksWorkflow.001.png

    472 -rw-r--r--@ 1 blyth  staff  241294 May 17  2020 Geant4OpticksWorkflow1.001.png      ## old workflow, with standard Geant4 to side
    496 -rw-r--r--@ 1 blyth  staff  251585 May 16  2020 Geant4OpticksWorkflow.001.png       ## old workflow, without standard Geant4 to side
    epsilon:Geant4OpticksWorkflow blyth$ 





Push to github::

    epsilon:Geant4OpticksWorkflow blyth$ cd ~/simoncblyth.bitbucket.io/
    epsilon:simoncblyth.bitbucket.io blyth$ git s
    On branch master
    Your branch is up-to-date with 'origin/master'.

    nothing to commit, working tree clean
    epsilon:simoncblyth.bitbucket.io blyth$ 
    epsilon:simoncblyth.bitbucket.io blyth$ 
    epsilon:simoncblyth.bitbucket.io blyth$ git push github 
    Counting objects: 11, done.
    Delta compression using up to 8 threads.
    Compressing objects: 100% (11/11), done.
    Writing objects: 100% (11/11), 1.01 KiB | 1.01 MiB/s, done.
    Total 11 (delta 9), reused 0 (delta 0)
    remote: Resolving deltas: 100% (9/9), completed with 7 local objects.
    To github.com:simoncblyth/simoncblyth.github.io.git
       dfc0daf..b9cc84f  master -> master
    epsilon:simoncblyth.bitbucket.io blyth$ 


