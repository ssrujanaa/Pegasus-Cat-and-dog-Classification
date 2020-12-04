#!/usr/bin/env python3
# coding: utf-8

import os
import logging
import pickle
from glob import glob
from pathlib import Path
from zipfile import ZipFile

import requests 
import pandas as pd

logging.basicConfig(level=logging.DEBUG)

from Pegasus.api import *

# --- Properties ---------------------------------------------------------------
props = Properties()
props["dagman.retry"] = "100"
props["pegasus.transfer.arguments"] = "-m 1"
props.write()

# --- Replica Catalog ----------------------------------------------------------
rc = ReplicaCatalog()

# checkpoint files
checkpoint_file = "checkpoint_file2.hdf5"
if not os.path.isfile(checkpoint_file):
    df = pd.DataFrame(list())
    df.to_csv(checkpoint_file)

rc.add_replica("local", checkpoint_file, Path(".").resolve() / checkpoint_file)
    
hpo_checkpoint_file = 'hpo_checkpoint.pkl'
if not os.path.isfile(hpo_checkpoint_file):
    df = pd.DataFrame(list())
    df.to_csv(hpo_checkpoint_file)

rc.add_replica("local", hpo_checkpoint_file, Path(".").resolve() / hpo_checkpoint_file)

# input images
input_files = [f for f in Path("./inputs").iterdir() if f.is_file() and f.name.endswith(".jpg")]
input_files.sort(key=lambda f: f.name.lower())
in_files=[]
 
for file in input_files:
    in_files.append(File(file.name))
    rc.add_replica(site="local", lfn=file.name, pfn=file.resolve())

rc.write()

# --- Transformation Catalog ---------------------------------------------------
# set to use OSG specific settings
RUN_ON_OSG = True

tools_container = Container("tools-container", Container.DOCKER, image="docker:///ssrujanaa/catsanddogs:latest")

pre_process_resize = Transformation( "preprocess1.py",
            site="condorpool",
            pfn="/usr/bin/preprocess1.py",
            is_stageable=False,
            container=None if RUN_ON_OSG else tools_container
            )

pre_process_augment = Transformation( "Augmentation.py",
            site="condorpool",
            pfn="/usr/bin/Augmentation.py",
            is_stageable=False,
            container=None if RUN_ON_OSG else tools_container
            )

data_split  = Transformation( "Data_Split.py",
            site="condorpool",
            pfn="/usr/bin/Data_Split.py",
            is_stageable=False,
            container=None if RUN_ON_OSG else tools_container
            )


hpo  = Transformation( "hpo_checkpointing.py",
            site="condorpool",
            pfn="/usr/bin/hpo_checkpointing.py",
            is_stageable=False,
            container=None if RUN_ON_OSG else tools_container
            )

vgg_model  = Transformation( "VGG_model.py",
            site="condorpool",
            pfn="/usr/bin/VGG_model.py",
            is_stageable=False,
            container=None if RUN_ON_OSG else tools_container
            )

test_model =  Transformation( "Test.py",
            site="condorpool",
            pfn="/usr/bin/Test.py",
            is_stageable=False,
            container=None if RUN_ON_OSG else tools_container
            )
                    
# add OSG specific settings
if RUN_ON_OSG:
    for tr in [pre_process_resize, pre_process_augment, data_split, hpo, vgg_model, test_model]:
        tr.add_condor_profile(requirements='HAS_SINGULARITY == True')
        tr.add_profiles(Namespace.CONDOR, key="+SingularityImage", value='"/cvmfs/singularity.opensciencegrid.org/ssrujanaa/catsanddogs:latest"')
    # add request_gpus for tr that need it

tc = TransformationCatalog()
tc.add_containers(tools_container)
tc.add_transformations(pre_process_resize,pre_process_augment,data_split,hpo,vgg_model,test_model)
tc.write()

# --- Workflow -----------------------------------------------------------------
wf = Workflow("Cats_and_Dogs", infer_dependencies=True)

resized_images = File('resized_images.txt')
all_files = [File("resized_{}".format(f.lfn)) for f in in_files]
labels = File('labels.txt')

job_preprocess1 = Job(pre_process_resize)\
                    .add_inputs(*in_files)\
                    .add_outputs(*all_files,resized_images,labels) 

aug_images_txt = File('augmentation.txt')
aug_labels_txt = File('aug_labels.txt')
augmented_files = []
for f in all_files:
    augmented_files.extend([File(str(f).replace("{}".format(os.path.splitext(str(f))[0]), "Aug_{}_{}".format(os.path.splitext(str(f))[0],i))) for i in range(3)])

    
job_preprocess2 = Job(pre_process_augment)\
                    .add_inputs(*all_files,labels)\
                    .add_outputs(aug_images_txt,aug_labels_txt,*augmented_files)

training_data = File('training.pkl')
testing_data = File('testing.pkl')
val_data = File('validation.pkl')

job_data_split = Job(data_split)\
                    .add_inputs(*augmented_files,labels)\
                    .add_outputs(training_data,testing_data,val_data)

model = File('model.h5')
output_file = File('hpo_results.pkl')
job_hpo = Job(hpo)\
                    .add_checkpoint(File(hpo_checkpoint_file), stage_out=True)\
                    .add_inputs(*augmented_files,training_data,testing_data,val_data)\
                    .add_profiles(Namespace.PEGASUS, key="maxwalltime", value=1)\
                    .add_outputs(output_file)

# clearing PYTHONPATH for this job makes this work on OSG
job_vgg_model = Job(vgg_model)\
                    .add_args("-epochs",6, "--batch_size",2)\
                    .add_checkpoint(File(checkpoint_file), stage_out=True)\
                    .add_inputs(*augmented_files,training_data,testing_data,val_data,output_file)\
                    .add_profiles(Namespace.PEGASUS, key="maxwalltime", value=1)\
                    .add_outputs(model)\
                    .add_env(PYTHONPATH="")

results_file = File('Result_Metrics.txt')
job_test_model = Job(test_model)\
                    .add_inputs(*augmented_files,testing_data,model)\
                    .add_outputs(results_file)

wf.add_jobs(job_preprocess1,job_preprocess2,job_data_split,job_hpo,job_vgg_model,job_test_model)                                    

try:
     wf.plan()
except PegasusClientError as e:
    print(e.output)
