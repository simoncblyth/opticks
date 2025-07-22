simoncblyth_github_io_index_not_updating
===========================================

Some mixup to debug, see nothing from 2025


* https://simoncblyth.github.io/?ver=10

* https://github.com/simoncblyth/simoncblyth.github.io/blob/master/index.html

* https://stackoverflow.com/questions/24851824/how-long-does-it-take-github-page-to-update-after-changing-index-html


Mon Jul 21
-------------

* push to github for simoncblyth.github.io updates the html in repo but in the html presentation


Tue Jul 22
--------------

* still not updating, checking deployments shows lots of fails but no hint of how to fix

* https://github.com/simoncblyth/simoncblyth.github.io/deployments

Last successful deployment was Feb 7th 2025

* https://github.com/simoncblyth/simoncblyth.github.io/commit/6302f7227ab3489ac991bbce7f1a93bf038d65b6

* https://github.com/simoncblyth/simoncblyth.github.io/settings/pages

* https://github.com/orgs/community/discussions/142349

* Settings > (GitHub) Pages > Build and deployment > Source

The source is "Deploy from a branch"

* changed that to use Actions after enabled actions for the repo

* https://github.com/simoncblyth/simoncblyth.github.io/settings/actions


First try failed with startup fail. Try changing permissions to
allow GitHub actions and make dummy commit.

* https://github.com/simoncblyth/simoncblyth.github.io/actions


Tue Jul 22 : Success : now the index is updated
---------------------------------------------------

* https://github.com/simoncblyth/simoncblyth.github.io/actions/runs/16433534596

* https://simoncblyth.github.io/



