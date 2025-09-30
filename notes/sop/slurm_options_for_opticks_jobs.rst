slurm_options_for_opticks_jobs
===============================

Normally exclusive GPU usage is required for sustained Opticks usage to prevent CUDA out-of-memory errors.
The below slurm option ensures exclusive access to a single GPU::

    --gpu-bind=single:1

Slurm probably implements this option by setting the CUDA_VISIBLE_DEVICES envvar
to control which GPUs each task can use.  Hence when using this option the manual
approach of setting the envvar should typically not be used.


