<a class="toc" id="table-of-contents"></a>

# ⚔🛡 Awesome Graph Adversarial Learning (Updating: 173 Papers)

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
<!-- ![](https://img.shields.io/github/stars/gitgiter/Graph-Adversarial-Learning)
![](https://img.shields.io/github/forks/gitgiter/Graph-Adversarial-Learning)
![](https://img.shields.io/github/license/gitgiter/Graph-Adversarial-Learning) -->

- [⚔🛡 Awesome Graph Adversarial Learning (Updating: 173 Papers)](#-awesome-graph-adversarial-learning-updating-173-papers)
- [⚔ Attack](#-attack)
  - [2021](#2021)
  - [2020](#2020)
  - [2019](#2019)
  - [2018](#2018)
  - [2017](#2017)
- [Taxonomies of Attack](#taxonomies-of-attack)
- [🛡 Defense](#-defense)
  - [2021](#2021-1)
  - [2020](#2020-1)
  - [2019](#2019-1)
  - [2018](#2018-1)
  - [2017](#2017-1)
- [🔐 Robustness Certification](#-robustness-certification)
- [⚖ Stability](#-stability)
- [🚀 Others](#-others)
- [📃 Survey](#-survey)
- [🔗 Resource](#-resource)
- [⚙ Toolbox](#-toolbox)

This repository contains Attack-related papers, Defense-related papers, Robustness Certification papers, etc., ranging from 2017 to 2021. All papers are available for download from [Latest Release](https://github.com/gitgiter/Graph-Adversarial-Learning/releases/latest).

If you find this repository useful, please cite:
*A Survey of Adversarial Learning on Graph, Arxiv'20*, [Link](https://arxiv.org/abs/2003.05730)

```bibtex
@article{chen2020survey,
  title={A Survey of Adversarial Learning on Graph},
  author={Chen, Liang and Li, Jintang and Peng, Jiaying and Xie, 
        Tao and Cao, Zengxu and Xu, Kun and He, Xiangnan and Zheng, Zibin},
  journal={arXiv preprint arXiv:2003.05730},
  year={2020}
}
```

<a class="toc" id ="1"></a>

# ⚔ Attack

[💨 Back to Top](#table-of-contents)

<a class="toc" id ="1-0"></a>

## 2021

<!-- ################################## -->
<details>
<summary>
<strong>Stealing Links from Graph Neural Networks</strong>
<a href="https://www.usenix.org/system/files/sec21summer_he.pdf"> 📝USENIX Security </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>Link Stealing Attacks</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Supervised/Unsupervised Training</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>GCN</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Link Prediction</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Traditional Link Prediction Algorithms</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>AUC</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>CiteSeer, Cora, Pubmed, AIDS, COX2, DHFR, ENZYMES, PROTEINS_full</td>
    </tr>
</table>
</details>

<a class="toc" id ="1-1"></a>

## 2020

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Attack on Community Detection by Hiding Individuals </strong>
<a href="https://arxiv.org/abs/2001.07933"> 📝WWW </a>
<a href="https://github.com/halimiqi/CD-ATTACK"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>CD-ATTACK</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Graph generation</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>GCN</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Community Detection</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN, Node2vec + K-means, ComE</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>DICE, MBA, RTA </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>Hiding performance measure M1 & M2</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>DBLP, Finance</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Manipulating Node Similarity Measures in Networks </strong>
<a href="https://arxiv.org/abs/1910.11529"> 📝AAMAS </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>FPTA</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td></td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td></td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Similarity</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>Node Similarity Measures</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Random, Greedy, High Jaccard Similarity (HJ)</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>Time</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Barabasi-Albert (BA), Erdos-Renyi (ER)</td>
    </tr>
</table>
</details>


<!-- ################################## -->
<details>
<summary>
<strong>A Restricted Black-box Adversarial Framework Towards Attacking Graph Embedding Models </strong>
<a href="https://arxiv.org/abs/1908.01297"> 📝AAAI </a>
<a href="https://github.com/SwiftieH/GFAttack"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>GF-Attack</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Graph signal processing</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td></td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN, SGC, DeepWalk, LINE</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Random, Degree, RL-S2V, <img src="http://latex.codecogs.com/gif.latex?A_{class}"></td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, Pubmed</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Indirect Adversarial Attacks via Poisoning Neighbors for Graph Convolutional Networks </strong>
<a href="https://arxiv.org/abs/2002.08012"> 📝BigData </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>POISONPROBE</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Binary search</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>GCN</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Nettack</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>ASR, Recall</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>CiteSeer, Cora-ML</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Non-target-specific Node Injection Attacks on Graph Neural Networks: A Hierarchical Reinforcement Learning Approach </strong>
<a href="http://faculty.ist.psu.edu/vhonavar/Papers/www20.pdf"> 📝WWW </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>NIPA</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Reinforcement learning, Nodes injection</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>GCN</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Random, FGA, Preferential attack</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora-ML, CiteSeer, Pubmed</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Attacks on Graph Neural Networks: Perturbations and their Patterns </strong>
<a href="https://dl.acm.org/doi/10.1145/3394520"> 📝TKDD </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>Fasttack</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Perturbations Impact Ranking</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>GCN</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN, CLN, DeepWalk</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Random, FGSM</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>Classification Margin, Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora-ML, CiteSeer, Polblogs, Pubmed</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>An Efficient Adversarial Attack on Graph Structured Data </strong>
<a href="https://www.aisafetyw.org/programme"> 📝IJCAI Workshop </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td></td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td></td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td></td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td></td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td></td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td></td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td></td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td></td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Practical Adversarial Attacks on Graph Neural Networks </strong>
<a href="https://grlplus.github.io/papers/8.pdf"> 📝ICML Workshop </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>GC-RWCS</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Greedy</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td></td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN, JKNetConcat, JKNetMaxpool</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Random, Degree, Betweenness, PageRank</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, Pubmed</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Link Prediction Adversarial Attack Via Iterative Gradient Attack </strong>
<a href="https://ieeexplore.ieee.org/abstract/document/9141291"> 📝IEEE Trans </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>IGA</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Gradient</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>GAE</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Link Prediction</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GAE, LRW, DeepWalk, Node2vec, CN, RA, Katz</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>RAN, DICE, GA</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>ASR, AML</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>NS, Yeast, FaceBook</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Attacks on Link Prediction Algorithms Based on Graph Neural Networks </strong>
<a href="https://iqua.ece.toronto.edu/papers/wlin-asiaccs20.pdf"> 📝Asia CCS </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>GGSP, OGSP</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Greedy</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td></td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Link Prediction</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>SEAL</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td></td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>ASR, AUC</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora-ML, CiteSeer, Pubmed</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial attack on BC classification for scale-free networks </strong>
<a href="https://aip.scitation.org/doi/10.1063/5.0003707"> 📝AIP Chaos </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>DALR, DILR</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Degree</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td></td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Network Structure</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>Broido and Clauset Classification</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>RLR</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Networks generated by BA and UCM</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Attackability Characterization of Adversarial Evasion Attack on Discrete Data </strong>
<a href="https://dl.acm.org/doi/10.1145/3394486.3403194"> 📝KDD </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>OMPGS</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Gradient Guided Greedy Search</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td></td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Classification on Sequential Discret Data</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>LSTM, LSTM-Sub, LSTM-Noise</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>SGS, FSGS, GradAttack, OMPGS-Rand</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>ANC, AI, SR (Attack Performance)</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>IPS, HER</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>MGA: Momentum Gradient Attack on Network</strong>
<a href="https://arxiv.org/abs/2002.11320"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>MGA</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Momentum gradient</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>GCN</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification, Community Detection</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN, DeepWalk, Node2vec, GraphGAN, LPA, Louvain</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>GradArgmax, RL-S2V, Nettack, FGA</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>ASR, AML</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, Polblogs</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Attacks to Scale-Free Networks: Testing the Robustness of Physical Criteria</strong>
<a href="https://arxiv.org/abs/2002.01249"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>RLR, DALR, DILR</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Random, Degree</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td></td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Network Structure</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>Physical Criteria</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td></td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>AML, (diagonal) distance, clustering coefficient</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Generated simplex networks</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Graph Universal Adversarial Attacks: A Few Bad Actors Ruin Graph Learning Models</strong>
<a href="https://arxiv.org/abs/2002.04784"> 📝Arxiv </a>
<a href="https://github.com/chisam0217/Graph-Universal-Attack"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>GUA</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Gradient</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>GCN</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN, DeepWalk, Node2Vec, GAT</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Random, VCA, FGA</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>ASR, AML</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, Polblogs</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Perturbations of Opinion Dynamics in Networks</strong>
<a href="https://arxiv.org/abs/2003.07010"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td></td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Graph Laplacian</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>Friedkin-Johnsen model</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Network Disruption</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td></td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Opinion dynamics model</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td></td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td></td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Network disruption: maximizing disagreement and polarization in social networks</strong>
<a href="https://arxiv.org/abs/2003.08377"> 📝Arxiv </a>
<a href="https://github.com/mayee107/network-disruption"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>Greedy et al.</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Greedy algorithm et al.</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>Friedkin-Johnsen model</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Network Disruption</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>Friedkin-Johnsen model</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td></td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>Disagreement,<br> Polarization</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Synthetic networks, Reddit, Twitter</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Scalable Attack on Graph Data by Injecting Vicious Nodes</strong>
<a href="https://arxiv.org/abs/2004.13825"> 📝ECML-PKDD </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>AFGSM</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Gradient</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>GCN</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN, GAT, DeepWalk</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Nettack, FGSM, Metattack</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>CiteSeer, Cora, DBLP, Pubmed, Reddit</td>
    </tr>
</table>
</details>


<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Attack on Hierarchical Graph Pooling Neural Networks|Gradient-Based Pooling Attack</strong>
<a href="https://arxiv.org/abs/2005.11560"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>Gradient-Based Pooling Attack</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Gradient</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>1-Layer HGP</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Graph Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>HGP, SAG, HGP-SL</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Random</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>DD, Mutagenicity, ER_MD, DHFR, AIDS, BZR</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Graph Backdoor</strong>
<a href="https://arxiv.org/abs/2006.11890"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>GTA</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Gradient</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td></td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification, Graph Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN, GraphSAGE, GAT</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td></td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>ASR, AMC, BAD, ADD</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Fingerprint, Malware, AIDS, Toxicant, Bitcoin, Facebook</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Backdoor Attacks to Graph Neural Networks|Subgraph-based Backdoor Attacks</strong>
<a href="https://arxiv.org/abs/2006.11165"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>Subgraph-based Backdoor Attacks</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Subgraph Generation</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td></td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Graph Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GIN</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Clean</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy, ASR</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Bitcoin, Twitter, COLLAB</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Attack on Large Scale Graph</strong>
<a href="https://arxiv.org/abs/2009.03488"> 📝Arxiv </a>
<a href="https://github.com/EdisonLeeeee/GraphAdv"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>SGA</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Gradient</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>SGC</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN, SGC, GAT, ClusterGCN, GraphSAGE</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>GradArgmax, Nettack</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>DAC, Accuracy, Classification Margin</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, Pubmed, Reddit</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Efficient Evasion Attacks to Graph Neural Networks via Influence Function</strong>
<a href="https://arxiv.org/abs/2009.00203"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>Influence-based Attack</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Influence Function</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td></td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN, SGC</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>OTA-KL, OTA-UL, Iter-KL, Iter-UL</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>ASR, Running Time</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, Pubmed</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Reinforcement Learning-based Black-Box Evasion Attacks to Link Prediction in Dynamic Graphs</strong>
<a href="https://arxiv.org/abs/2009.00163"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>RL-based Attack</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Reinforcement Learning</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td></td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Link Prediction</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>DyGCN</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Random-whole, Random-partial</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>F1</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Haggle, Hypertext, Trapping</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Semantic-preserving Reinforcement Learning Attack Against Graph Neural Networks for Malware Detection</strong>
<a href="https://arxiv.org/abs/2009.05602"> 📝Arxiv </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adaptive Adversarial Attack on Graph Embedding via GAN</strong>
<a href="https://link.springer.com/chapter/10.1007/978-981-15-9031-3_7"> 📝SocialSec </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Scalable Adversarial Attack on Graph Neural Networks with Alternating Direction Method of Multipliers</strong>
<a href="https://arxiv.org/abs/2009.10233"> 📝Arxiv </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>One Vertex Attack on Graph Neural Networks-based Spatiotemporal Forecasting</strong>
<a href="https://openreview.net/forum?id=W0MKrbVOxtd"> 📝ICLR OpenReview </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Single-Node Attack for Fooling Graph Neural Networks </strong>
<a href="https://openreview.net/forum?id=u4WfreuXxnk"> 📝ICLR OpenReview </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Black-Box Adversarial Attacks on Graph Neural Networks as An Influence Maximization Problem </strong>
<a href="https://openreview.net/forum?id=sbyjwhxxT8K"> 📝ICLR OpenReview </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Attacks on Deep Graph Matching </strong>
<a href="https://papers.nips.cc/paper/2020/file/ef126722e64e98d1c33933783e52eafc-Paper.pdf"> 📝NeurIPS </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Black-Box Adversarial Attacks on Graph Neural Networks with Limited Node Access </strong>
<a href="https://arxiv.org/abs/2006.05057"> 📝NeurIPS </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>A Graph Matching Attack on Privacy-Preserving Record Linkage </strong>
<a href="https://dl.acm.org/doi/abs/10.1145/3340531.3411931"> 📝CIKM </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Cross Entropy Attack on Deep Graph Infomax </strong>
<a href="https://ieeexplore.ieee.org/document/9180817"> 📝IEEE ISCAS </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Model Extraction Attacks on Graph Neural Networks: Taxonomy and Realization </strong>
<a href="https://arxiv.org/abs/2010.12751"> 📝Arxiv </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Learning to Deceive Knowledge Graph Augmented Models via Targeted Perturbation </strong>
<a href="https://arxiv.org/abs/2010.12872"> 📝ICLR OpenReview </a>
<a href="https://github.com/INK-USC/deceive-KG-models"> :octocat:Code </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Towards More Practical Adversarial Attacks on Graph Neural Networks </strong>
<a href="https://arxiv.org/abs/2006.05057"> 📝NeurIPS </a>
<a href="https://github.com/Mark12Ding/GNN-Practical-Attack"> :octocat:Code </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Label-Flipping Attack and Defense for Graph Neural Networks </strong>
📝ICDM 
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Exploratory Adversarial Attacks on Graph Neural Networks </strong>
📝ICDM
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Attacking Graph-Based Classification without Changing Existing Connections </strong>
<a href="https://cse.sc.edu/~zeng1/papers/2020-acsac-graph.pdf"> 📝ACSAC </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>A Targeted Universal Attack on Graph Convolutional Network </strong>
<a href="https://arxiv.org/abs/2011.14365"> 📝Arxiv </a>
<a href="https://github.com/Nanyuu/TUA"> :octocat:Code </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Query-free Black-box Adversarial Attacks on Graphs </strong>
<a href="https://arxiv.org/abs/2012.06757"> 📝Arxiv </a>
</summary>
</details>



<a class="toc" id ="1-2"></a>

## 2019

<!-- ################################## -->
<details>
<summary>
<strong>A Unified Framework for Data Poisoning Attack to Graph-based Semi-supervised Learning </strong>
<a href="https://arxiv.org/abs/1910.14147"> 📝NeurIPS </a>
<a href="https://github.com/xuanqing94/AdvSSL"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>G-SSL</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Gradient based asymptotic linear algorithm</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td></td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Classification, Regression</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>Label propagation & regularization algs</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Random, PageRank, Degree </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>Error rate, RMSE</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>cadata, E2006, mnist17, rcv1</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Examples on Graph Data: Deep Insights into Attack and Defense </strong>
<a href="https://arxiv.org/abs/1903.01610"> 📝IJCAI </a>
<a href="https://github.com/stellargraph/stellargraph/tree/develop/demos/interpretability"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>IG-FGSM, IG-JSMA</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Gradient</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>GCN</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>FGSM, JSMA, Nettack </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>Classification Margin, Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, PolBlogs</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective </strong>
<a href="https://arxiv.org/abs/1906.04214"> 📝IJCAI </a>
<a href="https://github.com/KaidiXu/GCN_ADV_Train"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>PGD, Min-Max</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Gradient</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>GCN</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>DICE, Metattack, Greedy </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>Misclassification  Rate</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Attacks on Graph Neural Networks via Meta Learning </strong>
<a href="https://arxiv.org/abs/1902.08412"> 📝ICLR </a>
<a href="https://github.com/danielzuegner/gnn-meta-attack"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>Metattack</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Gradient</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>GCN</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN, CLN, DeepWalk</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>DICE, Nettack, First-order</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>Misclassification Rate, Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, PolBlogs, PubMed</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>αCyber: Enhancing Robustness of Android Malware Detection System against Adversarial Attacks on Heterogeneous Graph based Model </strong>
<a href="https://dl.acm.org/doi/10.1145/3357384.3357875"> 📝CIKM </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>HG-Attack</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Label propagation algorithm, Nodes injection</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td></td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Malware Detection</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>Orig-HGC</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>AN-Attack </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>TP, TN, FP, FN, F1, Precision, Recall, Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Tencent Security Lab Dataset</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Data Poisoning Attack against Knowledge Graph Embedding </strong>
<a href="https://arxiv.org/abs/1904.12052"> 📝IJCAI </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td></td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Knowledge embedding</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td></td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Fact Plausibility Prediction</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>TransE, TransR, RESCAL</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>RA </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>MRR, HR@K</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>FB15k, WN18</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>GA Based Q-Attack on Community Detection</strong>
<a href="https://arxiv.org/abs/1811.00430"> 📝TCSS </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>Q-Attack</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Genetic algorithm</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td></td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Community Detection</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>FN, Lou, SOA, LPA, INF, Node2vec+KM</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Random, CDA, DBA </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>Modularity Q, NMI</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Karate, Dolphins, Football, Polbooks</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Attacking Graph-based Classification via Manipulating the Graph Structure </strong>
<a href="https://arxiv.org/abs/1903.00553"> 📝CCS </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td></td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td></td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>LinLBP</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification, Evasion</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>LinLBP, JWP, LBP, RW, LINE, DeepWalk, Node2vec, GCN</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Random, Nettack</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>FNR, FPR</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Facebook, Enron, Epinions, Twitter, Google+</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Attacks on Node Embeddings via Graph Poisoning </strong>
<a href="https://arxiv.org/abs/1809.01093"> 📝ICML </a>
<a href="https://github.com/abojchevski/node_embedding_attack"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td><img src="http://latex.codecogs.com/gif.latex?\mathcal{A}_{DW2}"> <img src="http://latex.codecogs.com/gif.latex?\mathcal{A}_{DW3}"></td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Gradient & Eigen-perturbation</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>DeepWalk</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification, Link Prediction</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>DeepWalk</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td><img src="http://latex.codecogs.com/gif.latex?\mathcal{B}_{rnd}"> <img src="http://latex.codecogs.com/gif.latex?\mathcal{B}_{eig}"> <img src="http://latex.codecogs.com/gif.latex?\mathcal{B}_{deg}"></td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>F1 Score, Classification Margin</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, PolBlogs</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Network Structural Vulnerability A Multi-Objective Attacker Perspective </strong>
<a href="https://ieeexplore.ieee.org/document/8275029"> 📝IEEE Trans </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Multiscale Evolutionary Perturbation Attack on Community Detection </strong>
<a href="https://arxiv.org/abs/1910.09741"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>EPA</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Genetic algorithm</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td></td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Community Detection</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GRE, INF, LOU</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td><img src="http://latex.codecogs.com/gif.latex?A_Q, A_S">, <img src="http://latex.codecogs.com/gif.latex?A_B, A_D,">  <img src="http://latex.codecogs.com/gif.latex?D_S, D_W"> </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>NMI, ARI</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Synthetic networks, Football, Email, Polblogs</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Time-aware Gradient Attack on Dynamic Network Link Prediction </strong>
<a href="https://arxiv.org/abs/1911.10561"> 📝IJCAI </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>TGA-Tra, TGA-Gre</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Gradient</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>DDNE</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Link Prediction</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>DDNE, ctRBM, GTRBM, dynAERNN</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Random, DGA, CNA </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>ASR, AML</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>RADOSLAW, LKML, FB-WOSN</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Attacking Graph Convolutional Networks via Rewiring </strong>
<a href="https://arxiv.org/abs/1906.03750"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>ReWatt</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Reinforcement Learning</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>GCN</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Graph Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>RL-S2V, RA </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>ASR</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>REDDIT-MULTI-12K, REDDIT-MULTI-5K, IMDB-MULTI</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Unsupervised Euclidean Distance Attack on Network Embedding </strong>
<a href="https://arxiv.org/abs/1905.11015"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>EDA</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Genetic algorithm</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>DeepWalk</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification, Community Detection</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>HOPE, LPA, EM, DeepWalk</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Random, DICE, RLS, DBA </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>NMI, Micro-F1, Macro-F1</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Karate, Game, Dolphin</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Generalizable Adversarial Attacks with Latent Variable Perturbation Modelling </strong>
<a href="https://arxiv.org/abs/1905.10864"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>DAGAER</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Generative model</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>VGAE</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Nettack </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>ASR</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Vertex Nomination, Consistent Estimation, and Adversarial Modification </strong>
<a href="https://arxiv.org/abs/1905.01776"> 📝Arxiv </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>PeerNets Exploiting Peer Wisdom Against Adversarial Attacks </strong>
<a href="https://arxiv.org/abs/1806.00088"> 📝ICLR (Poster) </a>
<a href="https://github.com/tantara/PeerNets-pytorch"> :octocat:Code </a>
</summary>
</details>




<a class="toc" id ="1-3"></a>

## 2018

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Attack on Graph Structured Data </strong>
<a href="https://arxiv.org/abs/1806.02371"> 📝ICML </a>
<a href="https://github.com/Hanjun-Dai/graph_adversarial_attack"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>RL-S2V, GradArgmax, GeneticAlg</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Reinforcement learning, Gradient, Genetic algorithm</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>GCN</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification, Graph Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN, GNN</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Random </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, PolBlogs, Finance</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Attacks on Neural Networks for Graph Data </strong>
<a href="https://arxiv.org/abs/1805.07984"> 📝KDD </a>
<a href="https://github.com/danielzuegner/nettack"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>Nettack</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Greedy search & gradient</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>GCN</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN, CLN, DeepWalk</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Rnd, FGSM </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>Classification Margin, Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora-ML, CiteSeer, PolBlogs</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Attacking Similarity-Based Link Prediction in Social Networks </strong>
<a href="https://arxiv.org/abs/1809.08368"> 📝AAMAS </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>Approx-Local</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Similarity methods</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td></td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Link Prediction</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>Local & Global similarity metrics</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Random, GreedyBase </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>Katz Similarity, ACT Distance, Similarity Score</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Random network, Facebook</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Hiding Individuals and Communities in a Social Network </strong>
<a href="https://arxiv.org/abs/1608.00375"> 📝Nature Human Behavior </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>DICE</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Disconnect Internally, Connect Externally</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td></td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td></td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td></td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td></td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td></td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td></td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Fake Node Attacks on Graph Convolutional Networks </strong>
<a href="https://arxiv.org/abs/1810.10751"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>Greedy, Greedy GAN</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Gradient</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>GCN, GAN</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>RA </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy, F1 Score, ASR </td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Attack Tolerance of Link Prediction Algorithms: How to Hide Your Relations in a Social Network </strong>
<a href="https://arxiv.org/abs/1809.00152"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>CTR OTC</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Neighbour score based on graph structure</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td></td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Link Prediction</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>Traditional Link Prediction Algs</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td></td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>AUC, AP</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>WTC 9/11, ScaleFree, Facebook, Random network</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Fast Gradient Attack on Network Embedding </strong>
<a href="https://arxiv.org/abs/1809.02797"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>FGA</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Gradient</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>GCN</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification, Community Detection</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN, GraRep, DeepWalk, Node2vec, LINE, GraphGAN</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Random, DICE, Nettack </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>ASR, AML </td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, PolBlogs</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Data Poisoning Attack against Unsupervised Node Embedding Methods </strong>
<a href="https://arxiv.org/abs/1810.12881"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>Opt-attack</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Gradient</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td>DeepWalk, LINE</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Link Prediction</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>DeepWalk, LINE, Node2vec, SC, GAE</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Random, PageRank, Degree sum, Shortest path </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>Similarity Score, AP </td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, Facebook</td>
    </tr>
</table>
</details>



<a class="toc" id ="1-4"></a>

## 2017

<!-- ################################## -->
<details>
<summary>
<strong>Practical Attacks Against Graph-based Clustering </strong>
<a href="https://arxiv.org/abs/1708.09056"> 📝CCS </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>Targeted noise injection, Small community attack</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Noise Injection</td>
    </tr>
    <tr>
        <!-- the surrogate model(s) used for attack -->
        <td><strong>Surrogate</strong></td>
        <td></td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Graph Clustering, Community Detection</td>
    </tr>
    <tr>
        <!-- the target model(s) to attack in this paper -->
        <td><strong>Target Model</strong></td>
        <td>SVD, Node2vec, Community Detection Algs</td>
        <!-- the baseline attack model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td></td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the attack in this paper -->
        <td><strong>Metric</strong></td>
        <td>ASR, FPR </td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Reverse Engineered DGA Domains, NXDOMAIN</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Sets for Regularising Neural Link Predictors </strong>
<a href="https://arxiv.org/abs/1707.07596"> 📝UAI </a>
<a href="https://github.com/uclmr/inferbeddings"> :octocat:Code </a>
</summary>
</details>

<a class="toc" id ="2"></a>

# Taxonomies of Attack
[💨 Back to Top](#table-of-contents)

![Attack](imgs/Attack.png)



<table align="center"><tr><td><table align="left"><tr><td>
<img  width = "420" src="imgs/attack_demo.png" ></a></td></tr></table></td><td><table align="left"
><tr><td>
<img width = "420"  src="imgs/p_and_e.png" ></a>
</td></tr></table></td></tr></table>


<a class="toc" id ="3"></a>

# 🛡 Defense

[💨 Back to Top](#table-of-contents)
<a class="toc" id ="3-0"></a>

## 2021

<!-- ################################## -->
<details>
<summary>
<strong>Learning to Drop: Robust Graph Neural Network via Topological Denoising </strong>
<a href="https://arxiv.org/abs/2011.07057"> 📝WSDM </a>
<a href="https://github.com/flyingdoog/PTDNet"> :octocat:Code </a>
</summary>
</details>

<a class="toc" id ="3-1"></a>

## 2020 

<!-- ################################## -->
<details>
<summary>
<strong>Transferring Robustness for Graph Neural Network Against Poisoning Attacks </strong>
<a href="https://arxiv.org/abs/1908.07558"> 📝WSDM </a>
<a href="https://github.com/tangxianfeng/PA-GNN"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>PA-GNN</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Penalized Aggregation, Meta Learning</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Structure Based</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GNN</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>GCN, GAT, GCN-Jaccard, RGCN, VPN </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Pubmed, Reddit, Yelp</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Power up! Robust Graph Convolutional Network against Evasion Attacks based on Graph Powering </strong>
<a href="https://arxiv.org/abs/1905.10029"> 📝ICLR OpenReview </a>
<a href="https://www.dropbox.com/sh/p36pzx1ock2iamo/AABEr7FtM5nqwC4i9nICLIsta?dl=0"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>r-GCN, VPN</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Graph Powering</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Objective Based</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>ManiReg, SemiEmb, LP, DeepWalk, ICA, Planetoid, GCN </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy, Robustness Merit, Attack Deterioration</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>CiteSeer, Cora, Pubmed</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>All You Need Is Low (Rank): Defending Against Adversarial Attacks on Graphs </strong>
<a href="https://dl.acm.org/doi/abs/10.1145/3336191.3371789"> 📝WSDM </a>
<a href="https://github.com/DSE-MSU/DeepRobust"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>GCN-SVD</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>SVD</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Preprocessing</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>GCN</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy, Classification Margin</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>CiteSeer, Cora-ML, PolBlogs</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>How Robust Are Graph Neural Networks to Structural Noise? </strong>
<a href="https://arxiv.org/abs/1912.10206"> 📝DLGMA </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td></td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Adversarial Training</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Adversarial Training</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GIN</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>GIN</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>F1 score</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Constructed graph</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Robust Detection of Adaptive Spammers by Nash Reinforcement Learning </strong>
<a href="https://arxiv.org/abs/2006.06069"> 📝KDD </a>
<a href="https://github.com/YingtongDou/Nash-Detect"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>Nash-Detect</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>A minimax game</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Detection Based</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Spam Detection</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td></td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Spam Detector</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Practical Effect, Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>YelpChi, YelpNYC, YelpZip</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Graph Structure Learning for Robust Graph Neural Networks </strong>
<a href="https://arxiv.org/abs/2005.10203"> 📝KDD </a>
<a href="https://github.com/DSE-MSU/DeepRobust"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>Pro-GNN</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Learns the graph structure and the GNN parameters simultaneously</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Hybrid</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>GAT, GCN-Jaccard, GCN-SVD </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, Polblogs, Pubmed</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Robust Graph Representation Learning via Neural Sparsification </strong>
<a href="https://proceedings.icml.cc/static/paper_files/icml/2020/2611-Paper.pdf"> 📝ICML </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>NeuralSparse</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Subgraphs Sampling </td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Preprocessing-based</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN, GraphSAGE, GAT, GIN</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>SS/RD, DropEdge, LDS </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Micro-F1,AUC, Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Reddit, PPI, Transaction, Cora, CiteSeer </td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>On The Stability of Polynomial Spectral Graph Filters </strong>
<a href="https://ieeexplore.ieee.org/abstract/document/9054072"> 📝ICASSP </a>
<a href="https://github.com/henrykenlay/spgf"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td></td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Polynomial graph filters</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Structure Based</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Graph signal processing</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GNN</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td></td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Laplacian distance</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Barabási-Albert, Sensor network</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Transferring Robustness for Graph Neural Network Against Poisoning Attacks </strong>
<a href="https://arxiv.org/abs/1908.07558"> 📝WSDM </a>
<a href="https://github.com/tangxianfeng/PA-GNN"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>PA-GNN</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Penalized Aggregation, Meta Learning</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Structure Based</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GNN</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>GCN, GAT, GCN-Jaccard, RGCN, VPN </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Pubmed, Reddit, Yelp</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>On the Robustness of Cascade Diffusion under Node Attacks </strong>
<a href="https://www.cs.au.dk/~karras/robustIC.pdf"> 📝WWW </a>
<a href="https://github.com/allogn/robustness"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td></td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>SEMR</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td></td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Cascade Diffusion</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>IC Model</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>NetShield </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>EMR, RNI, RIM</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Blogs, Minnesota, VK, Advogato, DBLP, BrightKite, ... </td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Friend or Faux: Graph-Based Early Detection of Fake Accounts on Social Networks </strong>
<a href="https://arxiv.org/abs/2004.04834"> 📝WWW </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td></td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>SybilEdge</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Preprocessing-based</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Fake Detection</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>Graph-based models</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>SybilRank, SybilBelief, SybilSCAR </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>AUC, RejectRate, SybilEdgeTR</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Facebook network</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Towards an Efficient and General Framework of Robust Training for Graph Neural Networks </strong>
<a href="https://arxiv.org/abs/2002.10947"> 📝ICASSP </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>GTA, ZO-GTA</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Greedy search, Zeroth-order</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Adversarial-based</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GNN</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>DICE, CE-PGD, CW-PGD </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Misclassification rate</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, PubMed</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Robust Graph Learning From Noisy Data </strong>
<a href="https://ieeexplore.ieee.org/abstract/document/8605364"> 📝IEEE Trans </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>RGC</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Graph regularization</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Prepocessing-based</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Clustering, Semisupervised Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>RPCA</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>SC, RKKM, RSC, SSR, CAN, TLSC </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy, NMI, Purity</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>YALE, JAFFE, ORL, TR41, TR45,  ...</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Robust Training of Graph Convolutional Networks via Latent Perturbation </strong>
<a href="https://www.cs.uic.edu/~zhangx/papers/JinZha20.pdf"> 📝ECML-PKDD </a>
<a href="https://github.com/tangxianfeng/PA-GNN"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>LAT-GCN</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Perturbing latent representations</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Structure Based</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification, Link prediction, Recommendation</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>GCN, ADV-GCN, MIN-MAX GCN, ...</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>CPU time, Accuracy, AUC, AP</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>CiteSeer, Cora, PubMed, MovieLens 100k</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters </strong>
<a href="https://arxiv.org/abs/2008.08692"> 📝CIKM </a>
<a href="https://github.com/safe-graph/DGFraud"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>CARE-GNN</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Reinforcement Learning</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Hybrid</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td></td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN, GAT, RGCN, GraphSAGE</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>GeniePath, Player2Vec, SemiGNN, GraphConsis </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>AUC, Recall</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Yelp, Amazon</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Provably Robust Node Classification via Low-Pass Message Passing </strong>
<a href="https://shenghua-liu.github.io/papers/icdm2020-provablerobust.pdf"> 📝ICDM </a>
  
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Dynamic Knowledge Graph-based Dialogue Generation with Improved Adversarial Meta-Learning </strong>
<a href="https://arxiv.org/abs/2004.08833"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>KDAD</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Adversarial Meta-learning</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Objective-based</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Dialogue Generation</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>Qadpt</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>TAware, Qadpt </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>BLEU, PPL, DISTINCT, ... </td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>HGZHZ</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Robust Collective Classification against Structural Attacks </strong>
<a href="http://www.auai.org/uai2020/proceedings/119_main_paper.pdf"> 📝Preprint </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>R-AMN</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Bound Analysis</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Objective-based</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>AMN</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Struct-RSAD </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy </td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Reuters, WebKB, Cora, CiteSeer</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Tensor Graph Convolutional Networks for Multi-relational and Robust Learning </strong>
<a href="https://arxiv.org/abs/2003.07729"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>TGCN</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Edge-dithering</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Processing-based</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification, Protein Prediction</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>GCN</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy, Macro F1 </td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, Pubmed, Polblogs, ...</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Topological Effects on Attacks Against Vertex Classification </strong>
<a href="https://arxiv.org/abs/2003.05822"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>StratDegree, GreedyCover</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>GreedyCover</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Processing-based</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Random Selection</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Required budget, Median margin </td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, Pubmed, Polblogs</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Evaluating Graph Vulnerability and Robustness using TIGER </strong>
<a href="https://arxiv.org/abs/2006.05648"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>TIGER</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td></td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Hybrid</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td></td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td></td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Average vertex betweenness, Spectral scaling, Effective resistance </td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>US power grid, Water Distribution Network</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Perturbations of Opinion Dynamics in Networks </strong>
<a href="https://arxiv.org/abs/2003.07010"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td></td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td></td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td></td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Network Disruption</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>Opinion dynamics models</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td></td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Polarization-disagreement index </td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td></td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>DefenseVGAE: Defending against Adversarial Attacks on Graph Data via a Variational Graph Autoencoder </strong>
<a href="https://arxiv.org/abs/2006.08900"> 📝Arxiv </a>
<a href="https://github.com/zhangao520/defense-vgae"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>DefenceVGAE</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>VGAE</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Processing-based</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>GCN-Jaccard, GCN-SVD, RGCN</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, PolBlogs</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>GNNGuard: Defending Graph Neural Networks against Adversarial Attacks </strong>
<a href="https://arxiv.org/abs/2006.08149"> 📝NeurIPS </a>
<a href="https://github.com/mims-harvard/GNNGuard"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>GNNGuard</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Network theory of homophily</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Structure-based</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN, GAT, GIN, ...</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>GNN-Jaccard, RobustGCN, GNN-SVD</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, ogbn-arxiv, DP</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Privacy Preserving Graph Embedding against Inference Attack </strong>
<a href="https://arxiv.org/abs/2008.13072"> 📝Arxiv </a>
<a href="https://github.com/uJ62JHD/Privacy-Preserving-Social-Network-Embedding">:octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>APDGE</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Adversarial Privacy-Purged</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Structure-based</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Privacy Protection</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GAE</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>GAE RM, CDSPIA </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td> Macro F1 </td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Yale, Rochester</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>RoGAT: a robust GNN combined revised GAT with adjusted graphs </strong>
<a href="https://arxiv.org/abs/2009.13038"> 📝Arxiv </a>
</summary>
</details>


<!-- ################################## -->
<details>
<summary>
<strong>Uncertainty-Matching Graph Neural Networks to Defend Against Poisoning Attacks </strong>
<a href="https://arxiv.org/abs/2009.14455"> 📝Arxiv </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>ResGCN: Attention-based Deep Residual Modeling for Anomaly Detection on Attributed Networks </strong>
<a href="https://arxiv.org/abs/2009.14738"> 📝Arxiv </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>A Novel Defending Scheme for Graph-Based Classification Against Graph Structure Manipulating Attack </strong>
<a href="https://link.springer.com/chapter/10.1007/978-981-15-9031-3_26"> 📝SocialSec </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Uncertainty-aware Attention Graph Neural Network for Defending Adversarial Attacks </strong>
<a href="https://arxiv.org/abs/2009.10235"> 📝Arxiv </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Iterative Deep Graph Learning for Graph Neural Networks: Better and Robust Node Embeddings </strong>
<a href="https://arxiv.org/abs/2006.13009"> 📝NeurIPS </a>
<a href="https://github.com/hugochan/IDGL"> :octocat:Code </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Towards Robust Graph Neural Networks against Label Noise </strong>
<a href="https://openreview.net/forum?id=H38f_9b90BO"> 📝ICLR OpenReview </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Graph Adversarial Networks: Protecting Information against Adversarial Attacks </strong>
<a href="https://openreview.net/forum?id=Q8ZdJahesWe"> 📝ICLR OpenReview </a>
<a href="https://github.com/liaopeiyuan/GAL"> :octocat:Code </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Ricci-GNN: Defending Against Structural Attacks Through a Geometric Approach </strong>
<a href="https://openreview.net/forum?id=_qoQkWNEhS"> 📝ICLR OpenReview </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Reliable Graph Neural Networks via Robust Aggregation </strong>
<a href="https://arxiv.org/abs/2010.15651"> 📝NeurIPS </a>
<a href="https://github.com/sigeisler/reliable_gnn_via_robust_aggregation"> :octocat:Code </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Graph Random Neural Networks for Semi-Supervised Learning on Graphs </strong>
<a href="https://arxiv.org/abs/2005.11079"> 📝NeurIPS </a>
<a href="https://github.com/Grand20/grand"> :octocat:Code </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Variational Inference for Graph Convolutional Networks in the Absence of Graph Data and Adversarial Settings </strong>
<a href="https://arxiv.org/abs/1906.01852"> 📝NeurIPS </a>
<a href="https://github.com/ebonilla/VGCN"> :octocat:Code </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Provable Overlapping Community Detection in Weighted Graphs </strong>
<a href="https://arxiv.org/abs/2004.07150"> 📝NeurIPS </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Community detection in sparse time-evolving graphs with a dynamical Bethe-Hessian </strong>
<a href="https://arxiv.org/abs/2006.04510"> 📝NeurIPS </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Node Copying for Protection Against Graph Neural Network Topology Attacks </strong>
<a href="https://arxiv.org/abs/2007.06704"> 📝Arxiv </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>A Feature-Importance-Aware and Robust Aggregator for GCN </strong>
<a href="https://dl.acm.org/doi/abs/10.1145/3340531.3411983"> 📝CIKM </a>
<a href="https://github.com/LiZhang-github/LA-GCN"> :octocat:Code </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Anti-perturbation of Online Social Networks by Graph Label Transition </strong>
<a href="https://arxiv.org/abs/2010.14121"> 📝Arxiv </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Graph Information Bottleneck </strong>
<a href="https://arxiv.org/abs/2010.12811"> 📝NeurIPS </a>
<a href="http://snap.stanford.edu/gib/"> :octocat:Code </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Detection on Graph Structured Data </strong>
<a href="https://dl.acm.org/doi/abs/10.1145/3411501.3419424"> 📝PPMLP </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Graph Contrastive Learning with Augmentations </strong>
<a href="https://arxiv.org/abs/2010.13902"> 📝NeurIPS </a>
<a href="https://github.com/Shen-Lab/GraphCL"> :octocat:Code </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Learning Graph Embedding with Adversarial Training Methods </strong>
<a href="https://arxiv.org/abs/1901.01250"> 📝IEEE Transactions on Cybernetics </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Unsupervised Adversarially-Robust Representation Learning on Graphs </strong>
<a href="https://arxiv.org/abs/2012.02486"> 📝Arxiv </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>I-GCN: Robust Graph Convolutional Network via Influence Mechanism </strong>
<a href="https://arxiv.org/abs/2012.06110"> 📝Arxiv </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversary for Social Good: Protecting Familial Privacy through Joint Adversarial Attacks </strong>
<a href="https://ojs.aaai.org//index.php/AAAI/article/view/6791"> 📝AAAI </a>
</summary>
</details>

<a class="toc" id ="3-2"></a>

## 2019

<!-- ################################## -->
<details>
<summary>
<strong>Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective </strong>
<a href="https://arxiv.org/abs/1906.04214"> 📝IJCAI </a>
<a href="https://github.com/KaidiXu/GCN_ADV_Train"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td></td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Adversarial Training</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Adversarial Training</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>GCN</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Misclassification Rate, Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Examples on Graph Data: Deep Insights into Attack and Defense </strong>
<a href="https://arxiv.org/abs/1903.01610"> 📝IJCAI </a>
<a href="https://github.com/DSE-MSU/DeepRobust"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>GCN-Jaccard</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Drop Edges</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Preprocessing</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>GCN</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Classification Margin, Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora-ML, CiteSeer, PolBlogs</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Investigating Robustness and Interpretability of Link Prediction via Adversarial Modifications </strong>
<a href="https://arxiv.org/abs/1905.00563"> 📝NAACL </a>
<a href="https://github.com/pouyapez/criage"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>CRIAGE</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Adversarial Modification</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Robustness Evaluation</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Link Prediction</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>Knowledge Graph Embedding</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td></td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Hits@K, MRR</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Nations, Kinship, WN18, YAGO3-10</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Robust Graph Convolutional Networks Against Adversarial Attacks </strong>
<a href="http://pengcui.thumedialab.com/papers/RGCN.pdf"> 📝KDD </a>
<a href="https://github.com/thumanlab/nrlweb/blob/master/static/assets/download/RGCN.zip"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>RGCN</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Gaussian-based Graph Convolution and Attention Mechanism</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Structure Based</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>GCN, GAT</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, Pubmed</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Virtual Adversarial Training on Graph Convolutional Networks in Node Classification </strong>
<a href="https://arxiv.org/abs/1902.11045"> 📝PRCV </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>SVAT, DVAT</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Virtual Adversarial Training</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Adversarial Training</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>GCN</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, Pubmed</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Comparing and Detecting Adversarial Attacks for Graph Deep Learning </strong>
<a href="https://rlgm.github.io/papers/57.pdf"> 📝RLGM@ICLR </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td></td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>KL Divergence</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Detection Based</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN, GAT</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td></td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Classification Margin, Accuracy, ROC, AUC</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, PolBlogs </td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Characterizing Malicious Edges targeting on Graph Neural Networks </strong>
<a href="https://arxiv.org/abs/1906.04214"> 📝ICLR OpenReview </a>
<a href="https://github.com/KaidiXu/GCN_ADV_Train"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>SL, OD, GGD,  LP+GGD, ENS</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Link Prediction, Subsampling, Neighbour Analysis</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Hybrid</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GNN, GCN </td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>LP</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>AUC</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Latent Adversarial Training of Graph Convolution Networks </strong>
<a href="https://graphreason.github.io/papers/35.pdf"> 📝LRGSD@ICML </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Batch Virtual Adversarial Training for Graph Convolutional Networks </strong>
<a href="https://arxiv.org/abs/1902.09192"> 📝ICML </a>
<a href="https://github.com/thudzj/BVAT"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>S-BVAT, O-BVAT</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>atch Virtual Adversarial Training</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Adversarial Training</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>LP, DeepWalk,  GAT, GPNN,  GCN, VAT, ...</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, Pubmed, Nell</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>αCyber: Enhancing Robustness of Android Malware Detection System against Adversarial Attacks on Heterogeneous Graph based Model </strong>
<a href="https://dl.acm.org/doi/10.1145/3357384.3357875"> 📝CIKM </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>Rad-HGC</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>HG-Defense</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Detection Based</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Malware Detection</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>Malware Detection System</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>FakeBank, CryptoMiner, AppCracked, MalFlayer, GameTrojan, BlackBaby, ...</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Detection Rate</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Tencent Security Lab Dataset</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Robustness of Similarity-Based Link Prediction </strong>
<a href="https://arxiv.org/abs/1909.01432"> 📝ICDM </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>IDOpt, IDRank</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Integer Program, Edge Ranking</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td></td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Link Prediction</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>Similarity-based Link Prediction Models</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>PPN</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>DPR</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>PA, PLD, TVShow, Gov</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>mproving Robustness to Attacks Against Vertex Classification </strong>
<a href="http://eliassi.org/papers/benmiller-mlg2019.pdf"> 📝MLG@KDD </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>SVM with a radial basis function kernel</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Augmented Feature, Edge Selecting</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Hybrid</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>SVM</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>GCN</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Classification Marigin</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Graph Adversarial Training: Dynamically Regularizing Based on Graph Structure </strong>
<a href="https://arxiv.org/abs/1902.08226"> 📝TKDE </a>
<a href="https://github.com/fulifeng/GraphAT"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>GCN-GATV</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>raph Adversarial Training, Virtual Adversarial Training</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Adversarial Training</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>LP, DeepWalk, SemiEmb, Planetoid, GCN, GraphSGAN</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, NELL</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Training Methods for Network Embedding </strong>
<a href="https://arxiv.org/abs/1908.11514"> 📝WWW </a>
<a href="https://github.com/wonniu/AdvT4NE_WWW2019"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>AdvT4NE</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Adversarial Training</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Adversarial Training</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Network embedding</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>Deepwalk</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>GF,DeepWalk, LINE,Node2vec, ...</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, Wiki, CA-GrQc, CA-HepTh</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>GraphDefense: Towards Robust Graph Convolutional Networks </strong>
<a href="https://arxiv.org/abs/1911.04429"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>GraphDefense</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Adversarial Training</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Adversarial Training</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>Drop Edges, Discrete Adversarial Training</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, Reddit</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Can Adversarial Network Attack be Defended? </strong>
<a href="https://arxiv.org/abs/1903.05994"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>Global-AT, Target-AT, SD, SCEL</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Adversarial Training, Smooth Defense</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Hybrid</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GNN</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>AT</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>ADR, ACD</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, PolBlogs</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Edge Dithering for Robust Adaptive Graph Convolutional Networks </strong>
<a href="https://arxiv.org/abs/1910.09590"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>AGCN</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Adaptive GCN with Edge Dithering</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Structure Based</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>GCN</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Accuracy</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, Pubmed, PolBlogs</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>GraphSAC: Detecting anomalies in large-scale graphs </strong>
<a href="https://arxiv.org/abs/1910.09589"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>GraphSVC</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Random, Consensus</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Detection Based</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Anomaly Detection</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>Anomaly Model</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>GAE, Amen, Radar, Degree, ...</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>AUC</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, Pubmed, PolBlogs</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Defense Framework for Graph Neural Network </strong>
<a href="https://arxiv.org/abs/1905.03679"> 📝Arxiv </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>DefNet</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>GAN, GER, ACL</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Hybrid</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Classification</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>GCN, GraphSAGE</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>GCN, GraphSAGE</td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Classification Margin</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Cora, CiteSeer, PolBlogs</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Graph Interpolating Activation Improves Both Natural and Robust Accuracies in Data-Efficient Deep Learning </strong>
<a href="https://arxiv.org/abs/1907.06800"> 📝Arxiv </a>
<a href="https://github.com/BaoWangMath/DNN-DataDependentActivation"> :octocat:Code </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Embedding: A robust and elusive Steganography and Watermarking technique </strong>
<a href="https://arxiv.org/abs/1912.01487"> 📝Arxiv </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Examining Adversarial Learning against Graph-based IoT Malware Detection Systems </strong>
<a href="https://arxiv.org/abs/1902.04416"> 📝Arxiv </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Target Defense Against Link-Prediction-Based Attacks via Evolutionary Perturbations </strong>
<a href="https://arxiv.org/abs/1809.05912"> 📝Arxiv </a>
</summary>
</details>

<a class="toc" id ="3-3"></a>

## 2018

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Personalized Ranking for Recommendation </strong>
<a href="https://dl.acm.org/citation.cfm?id=3209981"> 📝SIGIR </a>
<a href="https://github.com/hexiangnan/adversarial_personalized_ranking"> :octocat:Code </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>APR, AMF</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Adversarial Training based on MF-BPR</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Adversarial Training</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Recommendation</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>MF-BPR</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>ItemPop, MF-BPR, CDAE, NeuMF, IRGAN </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>HR, NDCG</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Yelp, Pinterest, Gowalla</td>
    </tr>
</table>
</details>

<a class="toc" id ="3-4"></a>

## 2017

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Sets for Regularising Neural Link Predictors </strong>
<a href="https://arxiv.org/abs/1707.07596"> 📝UAI </a>
<a href="https://github.com/uclmr/inferbeddings"> :octocat:Code </a>
</summary>
</details>

<a class="toc" id ="4"></a>

# 🔐 Robustness Certification

[💨 Back to Top](#table-of-contents)

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Immunization for Improving Certifiable Robustness on Graphs </strong>
<a href="https://arxiv.org/abs/2007.09647"> 📝WSDM'21 </a>
</summary>
</details>


<!-- ################################## -->
<details>
<summary>
<strong>Improving the Robustness of Wasserstein Embedding by Adversarial PAC-Bayesian Learning </strong>
<a href="http://staff.ustc.edu.cn/~hexn/papers/aaai20-adversarial-embedding.pdf"> 📝AAAI'20 </a>
</summary>
<table>
    <tr>
        <!-- the name of model(s) proposed in this paper -->
        <td><strong>Model</strong></td>
        <td>RAWEN</td>
        <!-- the core idea(s) of this paper -->
        <td><strong>Algorithm</strong></td>
        <td>Adversarial PAC-Bayesian learning</td>
    </tr>
    <tr>
        <!-- the defense type of the proposed defense model -->
        <td><strong>Defense Type</strong></td>
        <td>Objective Based</td>
        <!-- the target task(s) considered in this paper -->
        <td><strong>Target Task</strong></td>
        <td>Node Embedding</td>
    </tr>
    <tr>
        <!-- the target model(s) to enhance its robustness in this paper -->
        <td><strong>Target Model</strong></td>
        <td>Wasserstein embedding</td>
        <!-- the baseline model(s) compared in this paper -->
        <td><strong>Baseline</strong></td>
        <td>GF, LINE, Node2vec, SDNE ... </td>
    </tr>
    <tr>
        <!-- the metric(s) used to evaluate the defense in this paper -->
        <td><strong>Metric</strong></td>
        <td>Presion, Recall, AUC, F1</td>
        <!-- the dataset(s) used in this paper -->
        <td><strong>Dataset</strong></td>
        <td>Wiki-Vote, Epinions, Google, Email,Wiki</td>
    </tr>
</table>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Collective Robustness Certificates	 </strong>
<a href="https://openreview.net/forum?id=ULQdiUTHe3y"> 📝ICLR'21 OpenReview </a>
</summary>
</details>


<!-- ################################## -->
<details>
<summary>
<strong>Certifying Robustness of Graph Laplacian Based Semi-Supervised Learning </strong>
<a href="https://openreview.net/forum?id=cQyybLUoXxc"> 📝ICLR'21 OpenReview </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Certified Robustness of Graph Convolution Networks for Graph Classification under Topological Attacks </strong>
<a href="https://www.cs.uic.edu/~zhangx/papers/Jinetal20.pdf"> 📝NeurIPS'20 </a>
<a href="https://github.com/RobustGraph/RoboGraph"> :octocat:Code </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Certified Robustness of Community Detection against Adversarial Structural Perturbation via Randomized Smoothing </strong>
<a href="https://arxiv.org/abs/2002.03421"> 📝WWW'20 </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Efficient Robustness Certificates for Discrete Data: Sparsity - Aware Randomized Smoothing for Graphs, Images and More </strong>
<a href="https://proceedings.icml.cc/book/2020/file/4f7b884f2445ef08da9bbc77b028722c-Paper.pdf"> 📝ICML'20 </a>
<a href="https://github.com/abojchevski/sparse_smoothing"> :octocat:Code </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Abstract Interpretation based Robustness Certification for Graph Convolutional Networks </strong>
<a href="http://ecai2020.eu/papers/31_paper.pdf"> 📝ECAI'20 </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Certifiable Robustness of Graph Convolutional Networks under Structure Perturbation </strong>
<a href="https://dl.acm.org/doi/10.1145/3394486.3403217"> 📝KDD'20 </a>
<a href="https://github.com/danielzuegner/robust-gcn-structure"> :octocat:Code </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Certified Robustness of Graph Classification against Topology Attack with Randomized Smoothing </strong>
<a href="https://arxiv.org/abs/2009.05872"> 📝NeurIPS'20 </a>
</summary>
</details>


<!-- ################################## -->
<details>
<summary>
<strong>Certified Robustness of Graph Neural Networks against Adversarial Structural Perturbation </strong>
<a href="https://arxiv.org/abs/2008.10715"> 📝Arxiv'20 </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Certifiable Robustness and Robust Training for Graph Convolutional Networks </strong>
<a href="https://arxiv.org/abs/1906.12269"> 📝KDD'19 </a>
<a href="https://www.kdd.in.tum.de/research/robust-gcn/"> :octocat:Code </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Certifiable Robustness to Graph Perturbations </strong>
<a href="http://papers.nips.cc/paper/9041-certifiable-robustness-to-graph-perturbations"> 📝NeurIPS'19 </a>
<a href="https://github.com/abojchevski/graph_cert"> :octocat:Code </a>
</summary>
</details>

<a class="toc" id ="5"></a>

# ⚖ Stability

[💨 Back to Top](#table-of-contents)

<!-- ################################## -->
<details>
<summary>
<strong>Graph and Graphon Neural Network Stability </strong>
<a href="https://arxiv.org/abs/2008.01767"> 📝Arxiv'20 </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>On the Stability of Graph Convolutional Neural Networks under Edge Rewiring </strong>
<a href="https://arxiv.org/abs/2010.13747"> 📝Arxiv'20 </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Stability of Graph Neural Networks to Relative Perturbations
 </strong>
<a href="https://ieeexplore.ieee.org/document/9054341"> 📝ICASSP'20 </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Graph Neural Networks: Architectures, Stability and Transferability </strong>
<a href="https://arxiv.org/abs/2008.01767"> 📝Arxiv'20 </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Stability Properties of Graph Neural Networks </strong>
<a href="https://arxiv.org/abs/1905.04497"> 📝Arxiv'19 </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Stability and Generalization of Graph Convolutional Neural Networks </strong>
<a href="https://arxiv.org/abs/1905.01004"> 📝KDD'19 </a>
<a href="https://github.com/raspberryice/ala-gcn"> :octocat:Code </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>When Do GNNs Work: Understanding and Improving Neighborhood Aggregation </strong>
<a href="https://www.ijcai.org/Proceedings/2020/181"> 📝IJCAI'19 Workshop </a>
<a href="https://github.com/raspberryice/ala-gcn"> :octocat:Code </a>
</summary>
</details>

<a class="toc" id ="6"></a>

# 🚀 Others

[💨 Back to Top](#table-of-contents)

<!-- ################################## -->
<details>
<summary>
<strong>FLAG: Adversarial Data Augmentation for Graph Neural Networks </strong>
<a href="https://arxiv.org/abs/2010.09891"> 📝Arxiv'20 </a>
<a href="https://github.com/devnkong/FLAG"> :octocat:Code </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Dynamic Knowledge Graph-based Dialogue Generation with Improved Adversarial Meta-Learning </strong>
<a href="https://arxiv.org/abs/2004.08833"> 📝Arxiv'20 </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Watermarking Graph Neural Networks by Random Graphs </strong>
<a href="https://arxiv.org/abs/2011.00512"> 📝Arxiv'20 </a>
</summary>
</details>

<a class="toc" id ="7"></a>

# 📃 Survey

[💨 Back to Top](#table-of-contents)

<!-- ################################## -->
<details>
<summary>
<strong>Graph Neural Networks Taxonomy, Advances and Trends </strong>
<a href="https://arxiv.org/abs/2012.08752"> 📝Arxiv'20 </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>A Survey of Adversarial Learning on Graph </strong>
<a href="https://arxiv.org/abs/2003.05730"> 📝Arxiv'20 </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Attacks and Defenses on Graphs: A Review and Empirical Study </strong>
<a href="https://arxiv.org/abs/2003.00653"> 📝Arxiv'20 </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Attacks and Defenses in Images, Graphs and Text: A Review </strong>
<a href="https://arxiv.org/abs/1909.08072"> 📝Arxiv'19 </a>
</summary>
</details>

<!-- ################################## -->
<details>
<summary>
<strong>Adversarial Attack and Defense on Graph Data: A Survey </strong>
<a href="https://arxiv.org/abs/1812.10528"> 📝Arxiv'18 </a>
</summary>
</details>



<a class="toc" id ="8"></a>

# 🔗 Resource
[💨 Back to Top](#table-of-contents)


+ **Awesome Adversarial Learning on Recommender System** [Link](https://github.com/EdisonLeeeee/RS-Adversarial-Learning)
+ **Awesome Graph Attack and Defense Papers** [Link](https://github.com/ChandlerBang/awesome-graph-attack-papers)
+ **Graph Adversarial Learning Literature** [Link](https://github.com/safe-graph/graph-adversarial-learning-literature)
+ **A Complete List of All (arXiv) Adversarial Example Papers** [🌐Link](https://nicholas.carlini.com/writing/2019/all-adversarial-example-papers.html)
+ **Adversarial Attacks and Defenses Frontiers, Advances and Practice**, *KDD'20 tutorial*, [🌐Link](https://sites.google.com/view/kdd-2020-attack-and-defense)

<a class="toc" id ="9"></a>

# ⚙ Toolbox
[💨 Back to Top](#table-of-contents)

+ **DeepRobust** [Link](https://github.com/DSE-MSU/DeepRobust): A PyTorch adversarial library for attack and defense methods on images and graphs.
+ **GraphAdv** [Link](https://github.com/EdisonLeeeee/GraphAdv): A TensorFlow-based library for adversarial attacks and defense methods on graph.
+ **GraphGallery** [Link](https://github.com/EdisonLeeeee/GraphGallery): A PyTorch and TensorFlow library for geometric graph (adversarial) learning.
