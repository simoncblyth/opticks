slurm_options_for_opticks_jobs
===============================

Normally exclusive GPU usage is required for sustained Opticks usage to prevent CUDA out-of-memory errors.
The below slurm option ensures exclusive access to a single GPU::

    --gpu-bind=single:1

Slurm probably implements this option by setting the CUDA_VISIBLE_DEVICES envvar
to control which GPUs each task can use.  Hence when using this option the manual
approach of setting the envvar should typically not be used.



Avoid logfile overwriting when using slurm array
----------------------------------------------------

The below slurm directive configures a job array of 200 total
tasks indexed from 1 to 200.  The "%20" arranges that only 20 run concurrently.

::

    #SBATCH --array=1-200%20


Arrange for a separate directory for each job in job array with::

     LOGDIR=$SLURM_ARRAY_TASK_ID
     mkdir -p $LOGDIR
     cd $LOGDIR
     ... command to invoke executable...




