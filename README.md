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

- [Defense](#defense)
    - [1] [Certifiable Robustness and Robust Training for Graph Convolutional Networks](#101)
    - [2] [Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective](#102)
    - [3] [Power up! Robust Graph Convolutional Network against Evasion Attacks based on Graph Powering](#103)
    - [4] [Adversarial Examples on Graph Data: Deep Insights into Attack and Defense](#104)
    - [5] [Adversarial Defense Framework for Graph Neural Network](#105)
    - [6] [Investigating Robustness and Interpretability of Link Prediction via Adversarial Modifications](#106)
    - [7] [Robust Graph Convolutional Networks Against Adversarial Attacks](#107)
    - [8] [Can Adversarial Network Attack be Defended?](#108)
    - [9] [Virtual Adversarial Training on Graph Convolutional Networks in Node Classification](#109)
    - [10] [Comparing and Detecting Adversarial Attacks for Graph Deep Learning](#110)
    - [11] [Graph Adversarial Training: Dynamically Regularizing Based on Graph Structure](#111)
    - [12] [Characterizing Malicious Edges targeting on Graph Neural Networks](#112)
    - [13] [Batch Virtual Adversarial Training for Graph Convolutional Networks](#113)
    
- [Baselines](#Baselines)

# Attack
|Venue|Title|Model|Algorithm|Attack Type|Target Task|Target Model|Baseline|Metric\*|Dataset|Code|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|ICLR<br>2019|<a class="toc" id ="1"></a>[[1]](https://arxiv.org/abs/1902.08412)<br>[🔙](#toc)|Meta-Self<br>Meta-Train|Meta Gradient based on GCN|placeholder|Node Classification|GCN<br>CLN<br>DeepWalk|DICE<br>NETTACK<br>First-order|Misclassification rate(+)|Cora<br>CiteSeer<br>PolBlogs|[Link](https://www.kdd.in.tum.de/research/gnn-meta-attack/)|
|ICML<br>2019|<a class="toc" id ="2"></a>[[2]](https://arxiv.org/abs/1809.01093)<br>[🔙](#toc)|$\mathcal{A}_{DW2}$<br>$\mathcal{A}_{DW3}$|Gradient based on random walk|placeholder|Node Classification<br>Link Prediction|DeepWalk|$\mathcal{B}_{rnd}$<br>$\mathcal{B}_{eig}$<br>$\mathcal{B}_{deg}$|F1 score change(+)<br>Classification margin(-)<br>|Cora<br>Citeseer<br>PolBlogs|[Link](https://www.kdd.in.tum.de/research/node_embedding_attack/)
|Arxiv<br>2019|<a class="toc" id ="3"></a>[[3]](https://arxiv.org/abs/1906.03750)<br>[🔙](#toc)|ReWatt|Reinforcement learning based on GCN|placeholder|Graph Classification|GCN|RL-S2V<br>RA<br>RA-S|ASR(+)|REDDIT-MULTI-12K<br>REDDIT-MULTI-5K<br>IMDB-MULTI|-|
|IJCAI<br>2019|<a class="toc" id ="4"></a>[[4]](https://arxiv.org/abs/1906.04214)<br>[🔙](#toc)|CE-PGD<br>CW-PGD|Gradient based|Poision<br>Evasion<br>Topology<br>White-box|Node Classification|GCN|DICE<br>Meta-Self<br>Greedy|Misclassification rate(+)|Cora<br>Citeseer|[Link](https://github.com/KaidiXu/GCN_ADV_Train)|
|Arxiv<br>2019|<a class="toc" id ="5"></a>[[5]](https://arxiv.org/abs/1905.11015)<br>[🔙](#toc)|EDA|Genetic algorithm based on DeepWalk|placeholder|Node Classification<br>Community Detection|HOPE<br>LPA<br>EM<br>DeepWalk|RA<br>DICE<br>RLS<br>DBA|NMI(-)<br>Micro-F1(-)<br>Macro-F1(-)|Karate<br>Game<br>Dolphin|-|
|Arxiv<br>2019|<a class="toc" id ="6"></a>[[6]](https://arxiv.org/abs/1905.10864)<br>[🔙](#toc)|DAGAER|Generative model based on VGAE|Evasion<br>Feature<br>White-box|Node Classification|GCN|NETTACK|ASR(+)|Cora<br>CiteSeer|-|
|Arxiv<br>2019|<a class="toc" id ="7"></a>[[7]](https://arxiv.org/abs/1904.12052)<br>[🔙](#toc)|-|Knowledge embedding|placeholder|Fact Plausibility Prediction|TransE<br>TransR<br>RESCAL|RA|MRR(-)<br>HR@10(-)|FB15k<br>WN18|-|
|Arxiv<br>2019|<a class="toc" id ="8"></a>[[8]](https://arxiv.org/abs/1903.00553)<br>[🔙](#toc)|-|Based on LinLBP|placeholder|Node Classification<br>Detection Evade|LinLBP<br>JWP<br>LBP<br>RW<br>LINE<br>DeepWalk<br>Node2vec<br>GCN|RA<br>NETTACK|FNR(+)<br>FPR(+)|Facebook<br>Enron<br>Epinions<br>Twitter<br>Google+|-|
|TCSS<br>2019|<a class="toc" id ="9"></a>[[9]](https://arxiv.org/abs/1811.00430)<br>[🔙](#toc)|Q-Attack|Genetic algorithm|placeholder|Community Detection|FN<br>Lou<br>SOA<br>LPA<br>INF<br>Node2vec+KM|RA<br>CDA<br>DBA|Modularity Q(-)<br>NMI(-)|Karate<br>Dolphins<br>Football<br>Polbooks|-|
|Arxiv<br>2018|<a class="toc" id ="10"></a>[[10]](https://arxiv.org/abs/1810.10751)<br>[🔙](#toc)|Greedy<br>Greedy GAN|Gradient based on GCN, GAN|placeholder|Node Classification|GCN|RA|Accuracy(-)<br>F1 score(-)<br>ASR(+)|Cora<br>CiteSeer|-
|Arxiv<br>2018|<a class="toc" id ="11"></a>[[11]](https://arxiv.org/abs/1809.00152)<br>[🔙](#toc)|CTR<br>OTC|Neighbour score based on graph structure|placeholder|Link Prediction|Traditional Link Prediction Algs|-|AUC(-)<br>AP(-)|WTC 9/11<br>ScaleFree<br>Facebook<br>Randomly-generated|-|
|Arxiv<br>2018|<a class="toc" id ="12"></a>[[12]](https://arxiv.org/abs/1810.01110)<br>[🔙](#toc)|IGA|Gradient based on GAE|Evasion<br>Topology<br>W & B -box|Link Prediction|GAE<br>DeepWalk<br>Node2vec<br>CN<br>RA<br>Katz<br>LRW|RAN<br>DICE<br>GA|ASR(+)<br>AML(-)|NS<br>Yeast<br>FaceBook|-|
|ICML<br>2018|<a class="toc" id ="13"></a>[[13]](https://arxiv.org/abs/1806.02371)<br>[🔙](#toc)|RL-S2V|Reinforcement Learning|placeholder|Node/Graph Classification|GCN<br>GNN|Random Sampling|Accuracy(-)|Citeseer<br>Cora<br>Pubmed<br>Finance|[Link](https://github.com/Hanjun-Dai/graph_adversarial_attack)
|KDD<br>2018|<a class="toc" id ="14"></a>[[14]](https://arxiv.org/abs/1805.07984)<br>[🔙](#toc)|Nettack|Based on GCN|Evasion<br>Poision<br>Topology<br>Feature<br>W & B -box|Node Classification|GCN<br>CLN<br>DeepWalk|Rnd<br>FGSM|Classification margin(-)<br>Accuracy(-)|Cora-ML<br>CiteSeer<br>PolBlogs|[Link](https://github.com/danielzuegner/nettack)|
|Arxiv<br>2018|<a class="toc" id ="15"></a>[[15]](https://arxiv.org/abs/1809.02797)<br>[🔙](#toc)|FGA|Gradient based on GCN|placeholder|Node Classification<br>Community Detection|GCN<br>GraRep<br>DeepWalk<br>Node2vec<br>LINE<br>GraphGAN|RA<br>DICE<br>NETTACK|ASR(+)<br>AML(-)|Cora<br>CiteSeer<br>PolBlogs|-|
|Arxiv<br>2018|<a class="toc" id ="16"></a>[[16]](https://arxiv.org/abs/1810.12881)<br>[🔙](#toc)|Opt-attack|Gradient based on DeepWalk, LINE|placeholder|Link Prediction|DeepWalk<br>LINE<br>Node2vec<br>SC<br>GAE|RA<br>PageRank<br>Degree sum<br>Shortest path|Similarity score change(+)<br>AP(-)|Facebook<br>Cora<BR>Citeseer|-|
|Arxiv<br>2018|<a class="toc" id ="17"></a>[[17]](https://arxiv.org/abs/1809.08368)<br>[🔙](#toc)|Approx-Local|Similarity methods|placeholder|Link Prediction|Local&Global similarity metrics|RandomDel<br>GreedyBase|Katz similarity<br>ACT distance<br>Similarity score|scale-free<br>Facebook|-|
|CCS<br>2017|<a class="toc" id ="18"></a>[[18]](https://arxiv.org/abs/1708.09056)<br>[🔙](#toc)|Targeted noise injection<br>--<br>Small community attack|Noise injection|placeholder|Graph Clustering<br>Community Detection|SVD<br>node2vec<br>Community Detection Algs|-|ASR(+)FPR(+)|Reverse Engineered DGA Domains<br>NXDOMAIN|-|

**\* (+) means higher is better, (-) means the opposite.**

# Defense

| Venue| Title | Model | Algorithm | Defense Type | Target Task | Target Model | Baseline | Metric | Dataset | Code | 
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| KDD 2019 | <a class="toc" id ="101"></a> [[1]](https://arxiv.org/abs/1906.12269) [🔙](#toc) | GNN (trained with RH-U) | Robustness Certification, Robust Training | Robust Training | Node Classification | GNN, GCN | GNN (trained with CE, RCE, RH) | Accuracy, Averaged worst-case Margin | Citeseer, Cora-ML, Pubmed | [Link](https://www.kdd.in.tum.de/research/robust-gcn/) |
| IJCAI 2019 | <a class="toc" id ="102"></a> [[2]](https://arxiv.org/abs/1906.04214) [🔙](#toc) | - | Robust Training | Robust Training | Node Classification | GNN, GCN | GCN | Misclassiﬁcation Rates | Cora, Citeseer | [Link](https://github.com/KaidiXu/GCN_ADV_Train) |
| Arxiv 2019 | <a class="toc" id ="103"></a> [[3]](https://arxiv.org/abs/1905.10029) [🔙](#toc) | r-GCN, VPN | Graph Powering | Pre-processing | Node Classification | GCN | ManiReg, SemiEmb, LP, DeepWalk, ICA, Planetoid, Vanilla GCN | Accuracy | Citeseer, Cora, Pubmed | - |
| IJCAI 2019 | <a class="toc" id ="104"></a> [[4]](https://arxiv.org/abs/1903.01610) [🔙](#toc) | - | Drop Edges | Pre-processing | Node Classification | GCN | GCN | Classfication Margin, Accuracy | Cora-ML, Citeseer, Polblogs | [Link](https://github.com/stellargraph/stellargraph/) |
| Arxiv 2019 | <a class="toc" id ="105"></a> [[5]](https://arxiv.org/abs/1905.03679) [🔙](#toc) | DefNet | GAN,<br>GER,<br>ACL | Structure Based | Node Classification | GCN, GraphSAGE | GCN, GraphSage | Classfication Margin | Cora, Citeseer, Polblogs | - |
| Arxiv 2019 | <a class="toc" id ="106"></a> [[6]](https://arxiv.org/abs/1905.00563) [🔙](#toc) | CRIAGE | Adversarial Modification | Robustness Evaluation | Link Prediction | Knowledge Graph Embedding | DistMult, ConvE | Hits@K, MRR | Nations, Kinship, WN18, YAGO3-10 | - |
| KDD 2019 | <a class="toc" id ="107"></a> [[7]](http://pengcui.thumedialab.com/papers/RGCN.pdf) [🔙](#toc) | RGCN | Gaussian-based Graph Convolution | Structure Based | Node Classification | GCN | GCN, GAT | Accuracy | Cora, Citeseer, Pubmed | - |
| Arxiv 2019 | <a class="toc" id ="108"></a> [[8]](https://arxiv.org/abs/1903.05994) [🔙](#toc) | Global-AT, Target-AT, SD, SCEL | Adversarial Training, Smooth Defense | Hybrid | Node Classification | GCN, DeepWalk, node2vec, Louvain | AT | ADR, ACD | PoLBlogs, Cora, Citeseer | - |
| Arxiv 2019 | <a class="toc" id ="109"></a> [[9]](https://arxiv.org/abs/1902.11045) [🔙](#toc) | SVAT, DVAT | Virtual Adversarial Training | Adversarial Training | Node Classification | GCN | GCN | Accuracy | Cora, Citeseer, Pubmed | - |
| RLGM@ICLR 2019 | <a class="toc" id ="110"></a> [[10]](https://rlgm.github.io/papers/57.pdf) [🔙](#toc) | - | KL Divergence | Structure Based | Node Classification | GCN, GAT, Nettack | GCN, BGCN, GAT | Classfication Margin, Accuracy,<br>ROC, AUC | Cora, Citeseer, Polblogs | - |
| Arxiv 2019 | <a class="toc" id ="111"></a> [[11]](https://arxiv.org/abs/1902.08226) [🔙](#toc) | GCN-GATV | Graph Adversarial Training, Virtual Adversarial Training | Adversarial Training | Node Classification | GCN | LP, DeepWalk, SemiEmb, Planetoid, GCN, GraphSGAN | Accuracy | Citeseer, Cora, NELL | - |
| OpenReview 2018 | <a class="toc" id ="112"></a> [[12]](https://openreview.net/forum?id=HJxdAoCcYX) [🔙](#toc) | SL, OD, GGD, LP+GGD, ENS | Link Prediction, Subsampling, Neighbour Analysis| Hybrid | Link Prediction | GNN, GCN | LP | AUC | Cora, Citeseer | - |
| ICML 2019 | <a class="toc" id ="113"></a> [[13]](https://arxiv.org/abs/1902.09192) [🔙](#toc) | S-BVAT, O-BVAT | Batch Virtual Adversarial Training | Adversarial Training | Node Classification | GCN | ManiReg, SemiEmb, LP, DeepWalk, Planetoid, Monet, GAT, GPNN, GCN, VAT | Accuracy | Cora, Citeseer, Pubmed, Nell | [Link](https://github.com/thudzj/BVAT) |

# Baselines
| Baseline | Venue | Paper | Code |
|:-:|:-:|:-:|:-:|
| DICE | Nature Human Behaviour 2018 | [Hiding individuals and communities in a social network](https://arxiv.org/abs/1608.00375) | [Link](https://github.com/JHL-HUST/HiCode) |
| Nettack | KDD 2018 | [Adversarial Attacks on Neural Networks for Graph Data](https://arxiv.org/abs/1805.07984) | [Link](https://github.com/danielzuegner/nettack) |
| First-order | ICML 2017 | [Model-agnostic meta-learning for fast adaptation of deep networks](https://dl.acm.org/citation.cfm?id=3305498) | [Link](https://github.com/cbfinn/maml) |
| RL-S2V |  | [Link]() | [Link]() |
| RA |  | [Link]() | [Link]() |
| RA-S |  | [Link]() | [Link]() |
| RLS |  | [Link]() | [Link]() |
| DBA |  | [Link]() | [Link]() |
| CDA |  | [Link]() | [Link]() |
| RAN |  | [Link]() | [Link]() |
| GA |  | [Link]() | [Link]() |
| Random Sampling |  | [Link]() | [Link]() |
| Rnd |  | [Link]() | [Link]() |
| FGSM |  | [Link]() | [Link]() |
| PageRank |  | [Link]() | [Link]() |
| Degree sum |  | [Link]() | [Link]() |
| Shortest path |  | [Link]() | [Link]() |
| RandomDel |  | [Link]() | [Link]() |
| GreedyBase |  | [Link]() | [Link]() |
| GNN |  | [Link]() | [Link]() |
| GCN | ICLR 2017 | [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) | [Link](https://github.com/tkipf/gcn) |
| ManiReg |  | [Link]() | [Link]() |
| SemiEmb |  | [Link]() | [Link]() |
| LP |  | [Link]() | [Link]() |
| Deepwalk | KDD 2014 | [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652) | [Link](https://github.com/phanein/deepwalk) |
| ICA |  | [Link]() | [Link]() |
| Planetoid |  | [Link]() | [Link]() |
| Vanilla GCN |  | [Link]() | [Link]() |
| GraphSage | NIPS 2017 | [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) | [Link]() |
| DistMult |  | [Link]() | [Link]() |
| ConvE |  | [Link]() | [Link]() |
| GAT |  | [Link]() | [Link]() |
| AT | ICLR 2015 | [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) | [Link](https://github.com/tensorflow/cleverhans) |
| BGCN |  | [Link]() | [Link]() |
| GraphSGAN |  | [Link]() | [Link]() |
| Monet |  | [Link]() | [Link]() |
| GPNN |  | [Link]() | [Link]() |
| VAT |  | [Link]() | [Link]() |