#!/bin/bash

sbatch train_job.sh -m 1024 -h 8 -l 8 -u 1024
sbatch train_job.sh -m 1024 -h 16 -l 8 -u 1024
sbatch train_job.sh -m 1024 -h 8 -l 16 -u 1024
sbatch train_job.sh -m 1024 -h 16 -l 12 -u 1024
sbatch train_job.sh -m 1024 -h 16 -l 16 -u 1024
sbatch train_job.sh -m 1024 -h 8 -l 8 -u 2048
sbatch train_job.sh -m 1024 -h 16 -l 8 -u 2048
sbatch train_job.sh -m 1024 -h 8 -l 16 -u 2048
sbatch train_job.sh -m 1024 -h 16 -l 12 -u 2048
sbatch train_job.sh -m 1024 -h 16 -l 16 -u 2048
sbatch train_job.sh -m 1024 -h 8 -l 8 -u 4096
sbatch train_job.sh -m 1024 -h 16 -l 8 -u 4096
sbatch train_job.sh -m 1024 -h 8 -l 16 -u 4096
sbatch train_job.sh -m 1024 -h 16 -l 12 -u 4096
sbatch train_job.sh -m 1024 -h 16 -l 16 -u 4096
sbatch train_job.sh -m 2048 -h 8 -l 8 -u 4096
sbatch train_job.sh -m 2048 -h 16 -l 8 -u 4096
sbatch train_job.sh -m 2048 -h 8 -l 16 -u 4096
sbatch train_job.sh -m 2048 -h 16 -l 12 -u 4096
sbatch train_job.sh -m 2048 -h 16 -l 16 -u 4096
sbatch train_job.sh -m 2048 -h 32 -l 12 -u 4096
sbatch train_job.sh -m 2048 -h 32 -l 16 -u 4096
sbatch train_job.sh -m 2048 -h 32 -l 8 -u 4096