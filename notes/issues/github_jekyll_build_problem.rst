github_jekyll_build_problem
=============================

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



TODO: try to avoid issue by getting rid of the symlink with changed refs or simply copies instead of soft links::

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





* https://github.com/orgs/community/discussions/9104



