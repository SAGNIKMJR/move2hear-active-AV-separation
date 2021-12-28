# Move2Hear: Active Audio-Visual Source Separation
This repository contains the PyTorch implementation of our **ICCV-21 paper** and the associated datasets: 

[Move2Hear: Active Audio-Visual Source Separation](http://vision.cs.utexas.edu/projects/move2hear)<br />
Sagnik Majumder, Ziad Al-Halah, Kristen Grauman<br />
The University of Texas at Austin, Facebook AI Research

Project website: [http://vision.cs.utexas.edu/projects/move2hear](http://vision.cs.utexas.edu/projects/move2hear)

## Abstract
We introduce the active audio-visual source separation problem, where an agent must move intelligently in order to better isolate the sounds coming from an object of interest in its environment. The agent hears multiple audio sources simultaneously (e.g., a person speaking down the hall in a noisy household) and it must use its eyes and ears to automatically separate out the sounds originating from a target object within a limited time budget. Towards this goal, we introduce a reinforcement learning approach that trains movement policies controlling the agent's camera and microphone placement over time, guided by the improvement in predicted audio separation quality. We demonstrate our approach in scenarios motivated by both augmented reality (system is already co-located with the target object) and mobile robotics (agent begins arbitrarily far from the target object). Using state-of-the-art realistic audio-visual simulations in 3D environments, we demonstrate our model's ability to find minimal movement sequences with maximal payoff for audio source separation.

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
