# Move2Hear: Active Audio-Visual Source Separation
This repository contains the PyTorch implementation of our **ICCV-21 paper** and the associated datasets: 

[Move2Hear: Active Audio-Visual Source Separation](http://vision.cs.utexas.edu/projects/move2hear)<br />
Sagnik Majumder, Ziad Al-Halah, Kristen Grauman<br />
The University of Texas at Austin, Facebook AI Research

Project website: [http://vision.cs.utexas.edu/projects/move2hear](http://vision.cs.utexas.edu/projects/move2hear)

## Abstract
We introduce the active audio-visual source separation problem, where an agent must move intelligently in order to better isolate the sounds coming from an object of interest in its environment. The agent hears multiple audio sources simultaneously (e.g., a person speaking down the hall in a noisy household) and it must use its eyes and ears to automatically separate out the sounds originating from a target object within a limited time budget. Towards this goal, we introduce a reinforcement learning approach that trains movement policies controlling the agent's camera and microphone placement over time, guided by the improvement in predicted audio separation quality. We demonstrate our approach in scenarios motivated by both augmented reality (system is already co-located with the target object) and mobile robotics (agent begins arbitrarily far from the target object). Using state-of-the-art realistic audio-visual simulations in 3D environments, we demonstrate our model's ability to find minimal movement sequences with maximal payoff for audio source separation.

## Dependencies
This code has been tested with ```python 3.6.12```, ```habitat-api 0.1.4```, ```habitat-sim 0.1.4``` and ```torch 1.4.0```. Additional python package requirements are available in ```requirements.txt```.   
  
First, install the required versions of [habitat-api](https://github.com/facebookresearch/habitat-lab), [habitat-sim](https://github.com/facebookresearch/habitat-sim) and [torch](https://pytorch.org/) inside a [conda](https://www.anaconda.com/) environment. 

Next, install the remaining dependencies either by 
```
pip3 install -r requirements.txt
``` 
or by parsing ```requirements.txt``` to get the names and versions of individual dependencies and install them individually.

## Datasets
Download the AAViSS-specific datasets from [this link](https://bit.ly/3r02Skb), extract the zip and put it under the project root. The extracted ```data``` directory should have 3 types of data
1. **audio_data**: the pre-processed and pre-normalized raw monaural audio waveforms for training and evaluation    
2. **passive_datasets**: the dataset (audio source and receiver pair spatial attributes) for pre-training of passive separators    
3. **active_datasets**: the dataset (episode specification) for training of Move2Hear policies   
    
Make a directory named ```sound_spaces``` and place it in the same directory as the one where the project root resides. Download the [SoundSpaces](https://github.com/facebookresearch/sound-spaces/blob/main/soundspaces/README.md) Matterport3D **binaural RIRs** and **metadata**, and extract them into directories named ```sound_spaces/binaural_rirs/mp3d``` and ```sound_spaces/metadata/mp3d```, respectively.    
     
Download the [Matterport3D](https://niessner.github.io/Matterport/) dataset, and cache the observations relevant for the SoundSpaces simulator using [this script](https://github.com/facebookresearch/sound-spaces/blob/main/scripts/cache_observations.py) from the [SoundSpaces repository](https://github.com/facebookresearch/sound-spaces). Use resolutions of ```128 x 128``` for both RGB and depth sensors. Place the cached observations for all scenes (.pkl files) in ```sound_spaces/scene_observations_new```.    
     
For further info about the structuring of the associated datasets, refer to ```audio_separation/config/default.py``` or the task configs.              

## Code
###### Pretraining    
```
CUDA_VISIBLE_DEVICES=0 python3 main.py --exp-config audio_separation/config/pretrain_passive.yaml --model-dir runs/passive_pretrain/PRETRAIN_DIRNAME --run-type train NUM_PROCESSES 1
```
###### Policy Training
8 GPU DDPPO training:    
1. **Near-Target** 
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -u -m torch.distributed.launch --use_env --nproc_per_node 8 main.py --exp-config audio_separation/config/train/nearTarget.yaml --model-dir runs/train/nearTarget/NEAR_TARGET_TRAIN_DIRNAME --run-type train NUM_PROCESSES 14
```    
2. **Far-Target**  
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -u -m torch.distributed.launch --use_env --nproc_per_node 8 main.py --exp-config audio_separation/config/train/farTarget.yaml --model-dir runs/train/farTarget/FAR_TARGET_TRAIN_DIRNAME --run-type train NUM_PROCESSES 14
```   

###### Validation    
Link checkpoints using ```scripts/search_for_checkpoint_thru_validation/link_ckpts_for_val.ipynb``` to search for best checkpoint on the basis of validation.    
1. **Near-Target**    
```
CUDA_VISIBLE_DEVICES=0 python3 main.py --exp-config audio_separation/config/val/nearTarget.yaml --model-dir runs/val/nearTarget/NEAR_TARGET_VAL_DIRNAME --run-type eval NUM_PROCESSES 1
```
2. **Far-Target**  
```
CUDA_VISIBLE_DEVICES=0 python3 main.py --exp-config audio_separation/config/val/farTarget.yaml --model-dir runs/val/farTarget/FAR_TARGET_VAL_DIRNAME --run-type eval NUM_PROCESSES 1
```

Search for best checkpoint using ```scripts/search_for_checkpoint_thru_validation/find_bestCkpt_lowestValSTFTLoss.ipynb```.   
     
For unheard sounds, use ```config/val/nearTarget_unheard.yaml``` or ```config/val/farTarget_unheard.yaml```, and use the corresponding validation directory.          


###### Testing        
1. **Near-Target**    
```
CUDA_VISIBLE_DEVICES=0 python3 main.py --exp-config audio_separation/config/test/nearTarget.yaml --model-dir runs/test/nearTarget/NEAR_TARGET_TEST_DIRNAME --run-type eval NUM_PROCESSES 1
```   
       
2. **Far-Target**    
Copy configs and best checkpoints from Near-Target and Far-Target into one single checkpoint using ```scripts/farTarget_eval/copy_individualCkptsNCfgs_switchPolicyEval.ipynb``` for switching policies during eval.    
```
CUDA_VISIBLE_DEVICES=0 python3 main.py --exp-config audio_separation/config/test/farTarget.yaml --model-dir runs/test/farTarget/FAR_TARGET_TEST_DIRNAME --run-type eval NUM_PROCESSES 1
```
     
Compute test metric (STFT l2 loss or SI-SDR) values using ```scripts/separated_audio_quality/compute_separation_qualtiy.ipynb```.      
      
For unheard sounds, use ```config/test/nearTarget_unheard.yaml``` or ```config/test/farTarget_unheard.yaml```, and use the corresponding test directory.     

## Citation
```
@InProceedings{Majumder_2021_ICCV,
    author    = {Majumder, Sagnik and Al-Halah, Ziad and Grauman, Kristen},
    title     = {Move2Hear: Active Audio-Visual Source Separation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {275-285}
}
```

# License
This project is released under the MIT license, as found in the LICENSE file.
