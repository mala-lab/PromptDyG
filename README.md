# PromptDyG
Official PyTorch implementation of ''PromptDyG: Test-Time Prompt Adaptation on Dynamic Graphs''
# Overview
In this paper, we propose a novel framework, PromptDyG, for discrete-time dynamic graph learning. Specifically, we adapt the live-update evaluation setting as graph snapshots emerge sequentially, and the models are evaluated in a continual manner. Existing works typically utilize static GNN models to represent each graph snapshot, and a temporal module is designed to capture temporal dynamics for subsequent snapshot prediction. However, these methods overlook the structural shifts between training and test snapshots, leading to inferior performance. To address this problem, we propose to enhance the generality of the learned model at each time step via unsupervised test-time prompt adaptation. To preserve the temporal dependencies, a lightweight learnable graph prompting mechanism is introduced while keeping the backbone frozen. Theoretical and empirical results demonstrate that the proposed PromptDyG effectively enlarges the similarity margin between positive and negative node pairs under structural shifts for subsequent graph snapshot predictions. Moreover, the proposed method could act as a plug-and-play module, consistently improving different dynamic graph backbones.

![image](https://github.com/mala-lab/PromptDyG/blob/main/figures/PromptDyG.png)

# Accessing Datasets
Please see the `get_public_data.sh` script for accessing publicly available datasets used in our paper from [Stanford Large Network Dataset Collection](http://snap.stanford.edu/data/index.html).

# Dependencies
We recorded our complete conda environment configuration: `environment.yml`.

# Run experiments:
    $ sh run.sh

To facilitate reproducibility, we provide a recording of the experiment execution process in **Run process**. This video demonstrates the procedure to run the code and reproduce the reported results.

# Acknowledgements
> The code is implemented based on: [ROLAND: Graph Learning Framework for Dynamic Graphs](https://github.com/snap-stanford/roland)

## 📖 Citation
    
If you find this work useful, please cite our paper:

```bibtex
@inproceedings{ai2026PromptDyG,
  title     = {PromptDyG: Test-Time Prompt Adaptation on Dynamic Graphs},
  author    = {Ai, Guoguo and Niu, Chaoxi and Yan, Hui and Zhou, Joey Tianyi and Ong, Yew Soon and Pang, Guansong},
  booktitle = {Proceedings of the 43th International Conference on Machine Learning (ICML)},
  year      = {2026}
}



