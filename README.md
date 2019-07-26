<a class="toc" id ="toc"></a>
# Table of Contents
+ [Attack](#attack)
    + [1] [Adversarial attacks on graph neural networks via meta learning](#1)
    + [2] [Adversarial Attacks on Node Embeddings via Graph Poisoning](#2)
    + [3] [Attacking Graph Convolutional Networks via Rewiring](#3)
    + [4] [Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective](#4)
    + [5] [Unsupervised Euclidean Distance Attack on Network Embedding](#5)
    + [6] [Generalizable Adversarial Attacks Using Generative Models](#6)
    + [7] [Data Poisoning Attack against Knowledge Graph Embedding](#7)
    + [8] [Attacking Graph-based Classification via Manipulating the Graph Structure](#8)
    + [9] [GA Based Q-Attack on Community Detection](#9)
    + [10] [Attack Graph Convolutional Networks by Adding Fake Nodes](#10)
    + [11] [Attack Tolerance of Link Prediction Algorithms: How to Hide Your Relations in a Social Network](#11)
    + [12] [Link Prediction Adversarial Attack](#12)
    + [13] [Adversarial Attack on Graph Structured Data](#13)
    + [14] [Adversarial Attacks on Neural Networks for Graph Data](#14)
    + [15] [Fast Gradient Attack on Network Embedding](#15)
    + [16] [Data Poisoning Attack against Unsupervised Node Embedding Methods](#16)
    + [17] [Attacking Similarity-Based Link Prediction in Social Networks](#17)
    + [18] [Practical Attacks Against Graph-based Clustering](#18)
<a class="toc" id ="attack"></a>
# Attack
|Venue|Title|Model|Algorithm|Target Task|Target Model|Baseline|Metric\*|Dataset|Code|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|ICLR<br>2019|<a class="toc" id ="1"></a>[[1]](https://arxiv.org/abs/1902.08412)<br>[🔙](#toc)|Meta-Self<br>Meta-Train|Meta Gradient based on GCN|Node Classification|GCN<br>CLN<br>DeepWalk|DICE<br>NETTACK<br>First-order|Misclassification rate(+)|Cora<br>CiteSeer<br>PolBlogs|[Link](https://www.kdd.in.tum.de/research/gnn-meta-attack/)|
|ICML<br>2019|<a class="toc" id ="2"></a>[[2]](https://arxiv.org/abs/1809.01093)<br>[🔙](#toc)|$\mathcal{A}_{DW2}$<br>$\mathcal{A}_{DW3}$|Gradient based on random walk|Node Classification<br>Link Prediction|DeepWalk|$\mathcal{B}_{rnd}$<br>$\mathcal{B}_{eig}$<br>$\mathcal{B}_{deg}$|F1 score change(+)<br>Classification margin(-)<br>|Cora<br>Citeseer<br>PolBlogs|[Link](https://www.kdd.in.tum.de/research/node_embedding_attack/)
|Arxiv<br>2019|<a class="toc" id ="3"></a>[[3]](https://arxiv.org/abs/1906.03750)<br>[🔙](#toc)|ReWatt|Reinforcement learning based on GCN|Graph Classification|GCN|RL-S2V<br>RA<br>RA-S|ASR(+)|REDDIT-MULTI-12K<br>REDDIT-MULTI-5K<br>IMDB-MULTI|-|
|IJCAI<br>2019|<a class="toc" id ="4"></a>[[4]](https://arxiv.org/abs/1906.04214)<br>[🔙](#toc)|CE-PGD<br>CW-PGD|Gradient based|Node Classification|GCN|DICE<br>Meta-Self attack<br>Greedy attack|Misclassification rate(+)|Cora<br>Citeseer|-|
|Arxiv<br>2019|<a class="toc" id ="5"></a>[[5]](https://arxiv.org/abs/1905.11015)<br>[🔙](#toc)|EDA|Genetic algorithm based on DeepWalk|Node Classification<br>Community Detection|HOPE<br>LPA<br>EM<br>DeepWalk|RA<br>DICE<br>RLS<br>DBA|NMI(-)<br>Micro-F1(-)<br>Macro-F1(-)|Karate<br>Game<br>Dolphin|-|
|Arxiv<br>2019|<a class="toc" id ="6"></a>[[6]](https://arxiv.org/abs/1905.10864)<br>[🔙](#toc)|DAGAER|Generative model based on GCN and MLP|Node Classification|GCN|NETTACK|ASR(+)|Cora<br>CiteSeer|-|
|Arxiv<br>2019|<a class="toc" id ="7"></a>[[7]](https://arxiv.org/abs/1904.12052)<br>[🔙](#toc)|-|Knowledge embedding|Fact Plausibility Prediction|TransE<br>TransR<br>RESCAL|RA|MRR(-)<br>HR@10(-)|FB15k<br>WN18|-|
|Arxiv<br>2019|<a class="toc" id ="8"></a>[[8]](https://arxiv.org/abs/1903.00553)<br>[🔙](#toc)|-|Based on LinLBP|Node Classification<br>Detection Evade|LinLBP<br>JWP<br>LBP<br>RW<br>LINE<br>DeepWalk<br>Node2vec<br>GCN|RA<br>NETTACK|FNR(+)<br>FPR(+)|Facebook<br>Enron<br>Epinions<br>Twitter<br>Google+|-|
|TCSS<br>2019|<a class="toc" id ="9"></a>[[9]](https://arxiv.org/abs/1811.00430)<br>[🔙](#toc)|Q-Attack|Genetic algorithm|Community Detection|FN<br>Lou<br>SOA<br>LPA<br>INF<br>Node2vec+KM|RA<br>CDA<br>DBA|Modularity Q(-)<br>NMI(-)|Karate<br>Dolphins<br>Football<br>Polbooks|-|
|Arxiv<br>2018|<a class="toc" id ="10"></a>[[10]](https://arxiv.org/abs/1810.10751)<br>[🔙](#toc)|Greedy<br>Greedy GAN|Gradient based on GCN, GAN|Node Classification|GCN|RA|Accuracy(-)<br>F1 score(-)<br>ASR(+)|Cora<br>CiteSeer|-
|Arxiv<br>2018|<a class="toc" id ="11"></a>[[11]](https://arxiv.org/abs/1809.00152)<br>[🔙](#toc)|CTR<br>OTC|Neighbour score based on graph structure|Link Prediction|Traditional Link Prediction Algs|-|AUC(-)<br>AP(-)|WTC 9/11<br>ScaleFree<br>Facebook<br>Randomly-generated|-|
|Arxiv<br>2018|<a class="toc" id ="12"></a>[[12]](https://arxiv.org/abs/1810.01110)<br>[🔙](#toc)|IGA|Gradient based on GAE|Link Prediction|GAE<br>DeepWalk<br>Node2vec<br>CN<br>RA<br>Katz<br>LRW|RAN<br>DICE<br>GA|ASR(+)<br>AML(-)|NS<br>Yeast<br>FaceBook|-|
|ICML<br>2018|<a class="toc" id ="13"></a>[[13]](https://arxiv.org/abs/1806.02371)<br>[🔙](#toc)|RL-S2V|Reinforcement Learning|Node/Graph Classification|GCN<br>GNN|Random Sampling|Accuracy(-)|Citeseer<br>Cora<br>Pubmed<br>Finance|[Link](https://github.com/Hanjun-Dai/graph_adversarial_attack)
|KDD<br>2018|<a class="toc" id ="14"></a>[[14]](https://arxiv.org/abs/1805.07984)<br>[🔙](#toc)|Nettack|Based on GCN|Node Classification|GCN<br>CLN<br>DeepWalk|Rnd<br>FGSM|Classification margin(-)<br>Accuracy(-)|Cora-ML<br>CiteSeer<br>PolBlogs|[Link](https://github.com/danielzuegner/nettack)|
|Arxiv<br>2018|<a class="toc" id ="15"></a>[[15]](https://arxiv.org/abs/1809.02797)<br>[🔙](#toc)|FGA|Gradient based on GCN|Node Classification<br>Community Detection|GCN<br>GraRep<br>DeepWalk<br>Node2vec<br>LINE<br>GraphGAN|RA<br>DICE<br>NETTACK|ASR(+)<br>AML(-)|Cora<br>CiteSeer<br>PolBlogs|-|
|Arxiv<br>2018|<a class="toc" id ="16"></a>[[16]](https://arxiv.org/abs/1810.12881)<br>[🔙](#toc)|Opt-attack|Gradient based on DeepWalk, LINE|Link Prediction|DeepWalk<br>LINE<br>Node2vec<br>SC<br>GAE|RA<br>PageRank<br>Degree sum<br>Shortest path|Similarity score change(+)<br>AP(-)|Facebook<br>Cora<BR>Citeseer|-|
|Arxiv<br>2018|<a class="toc" id ="17"></a>[[17]](https://arxiv.org/abs/1809.08368)<br>[🔙](#toc)|Approx-Local|Similarity methods|Link Prediction|Local&Global similarity metrics|RandomDel<br>GreedyBase|Katz similarity<br>ACT distance<br>Similarity score|scale-free<br>Facebook|-|
|CCS<br>2017|<a class="toc" id ="18"></a>[[18]](https://arxiv.org/abs/1708.09056)<br>[🔙](#toc)|Targeted noise injection<br>Small community attack|Noise injection|Graph Clustering<br>Community Detection|SVD<br>node2vec<br>Community Detection Algs|-|ASR(+)FPR(+)|Reverse Engineered DGA Domains<br>NXDOMAIN|-|
**\* (+) means higher is better, (-) means the opposite.**






+ White box attack: The attacker has access to the model’s parameters
+ Black box attack: The attacker has no access to these parameters, i.e., it uses a different model or no model at all to generate adversarial network with the hope that these will transfer to the target model


