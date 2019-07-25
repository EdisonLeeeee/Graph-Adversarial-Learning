<a class="toc" id ="toc"></a>
# Table of Contents
+ [Attack](#attack)
    + [1] [Adversarial attacks on graph neural networks via meta learning](#1)
    + [2] [Adversarial Attacks on Node Embeddings via Graph Poisoning](#2)
    + [3] [Attacking Graph Convolutional Networks via Rewiring](#3)
    + [4] [Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective](#4)
    + [5] [Link Prediction Adversarial Attack](#5)
    + [6] [Adversarial Attack on Graph Structured Data](#6)
    + [7] [Adversarial Attacks on Neural Networks for Graph Data](#7)
    + [8] [Fast Gradient Attack on Network Embedding](#8)
    + [9] [Data Poisoning Attack against Unsupervised Node Embedding Methods](#9)
<a class="toc" id ="attack"></a>
# Attack
|Venue|Title|Model|Algorithm|Target Task|Target Model|Baseline|Metric|Dataset|Code|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|ICLR<br>2019|<a class="toc" id ="1"></a>[[1]](https://arxiv.org/abs/1902.08412)<br>[🔙](#toc)|Meta-Self<br>Meta-Train|Meta Gradient based on GCN|Node classification|GCN<br>CLN<br>DeepWalk|DICE<br>NETTACK<br>First-order|Misclassification rate(+)|Cora<br>CiteSeer<br>PolBlogs|[Link](https://www.kdd.in.tum.de/research/gnn-meta-attack/)|
|ICML<br>2019|<a class="toc" id ="2"></a>[[2]](https://arxiv.org/abs/1809.01093)<br>[🔙](#toc)|$\mathcal{A}_{DW2}$<br>$\mathcal{A}_{DW3}$|Gradient based on random walk|Node classification<br>Link prediction|DeepWalk|$\mathcal{B}_{rnd}$<br>$\mathcal{B}_{eig}$<br>$\mathcal{B}_{deg}$|F1 score change(+)<br>Classification margin(-)<br>|Cora<br>Citeseer<br>PolBlogs|[Link](https://www.kdd.in.tum.de/research/node_embedding_attack/)
|Arxiv<br>2019|<a class="toc" id ="3"></a>[[3]](https://arxiv.org/abs/1906.03750)<br>[🔙](#toc)|ReWatt|Reinforcement learning based on GCN|Graph classification|GCN|RL-S2V<br>Random<br>Random-S|Success rate(+)|REDDIT-MULTI-12K<br>REDDIT-MULTI-5K<br>IMDB-MULTI|None|
|IJCAI<br>2019|<a class="toc" id ="4"></a>[[4]](https://arxiv.org/abs/1906.04214)<br>[🔙](#toc)|CE-PGD<br>CW-PGD|Gradient based|Node classification|GCN|DICE<br>Meta-Self attack<br>Greedy attack|Misclassification rate(+)|Cora<br>Citeseer|None|
|Arxiv<br>2018|<a class="toc" id ="5"></a>[[5]](https://arxiv.org/abs/1810.01110)<br>[🔙](#toc)|IGA|Gradient based on GAE|Link prediction|GAE<br>DeepWalk<br>node2vec<br>CN<br>RA<br>Katz<br>LRW|RAN<br>DICE<br>GA|ASR(+)<br>AML(-)|NS<br>Yeast<br>FaceBook|None
|ICML<br>2018|<a class="toc" id ="6"></a>[[6]](https://arxiv.org/abs/1806.02371)<br>[🔙](#toc)|RL-S2V|Reinforcement Learning|Graph classification|GCN|Random Sampling|Accuracy(-)|Citeseer<br>Cora<br>Pubmed<br>Finance|[Link](https://github.com/Hanjun-Dai/graph_adversarial_attack)
|SIGKDD<br>2018|<a class="toc" id ="7"></a>[[7]](https://arxiv.org/abs/1805.07984)<br>[🔙](#toc)|Nettack|Based on GCN|Node classification|GCN<br>CLN<br>DeepWalk|Rnd<br>FGSM|Classification margin(-)<br>Accuracy(-)|Cora-ML<br>CiteSeer<br>PolBlogs|[Link](https://github.com/danielzuegner/nettack)|
|Arxiv<br>2018|<a class="toc" id ="8"></a>[[8]](https://arxiv.org/abs/1809.02797)<br>[🔙](#toc)|FGA|Gradient based on GCN|Node classification<br>Community detection|GCN<br>GraRep<br>DeepWalk<br>Node2vec<br>LINE<br>GraphGAN|RA<br>DICE<br>NETTACK|ASR(+)<br>AML(-)|Cora<br>CiteSeer<br>PolBlogs|None|
|Arxiv<br>2018|<a class="toc" id ="9"></a>[[9]](https://arxiv.org/abs/1810.12881)<br>[🔙](#toc)|Opt-attack|Gradient based on DeepWalk,  LINE|Link prediction|DeepWalk<br>LINE<br>Node2vec<br>SC<br>GAE|RA<br>PageRank<br>Degree sum<br>Shortest path|Similarity score change(+)<br>AP(-)|Facebook<br>Cora<BR>Citeseer|None|



