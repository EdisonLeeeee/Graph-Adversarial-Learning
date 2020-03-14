# Table of Contents

<a class="toc" id ="toc1"></a>

+ [Attack](#attack)
    + [1] [MGA: Momentum Gradient Attack on Network](#1)
    + [2] [Adversarial Attacks to Scale-Free Networks: Testing the Robustness of Physical Criteria](#2)
    + [3] [Graph Universal Adversarial Attacks: A Few Bad Actors Ruin Graph Learning Models](#3)
    + [4] [Adversarial Attack on Community Detection by Hiding Individuals](#4)
    + [5] [Manipulating Node Similarity Measures in Networks](#5)
    + [6] [Node Injection Attacks on Graphs via Reinforcement Learning](#6)
    + [7] [Vertex Nomination, Consistent Estimation, and Adversarial Modification](#7)
    + [8] [The General Black-box Attack Method for Graph Neural Networks](#8)
    + [9] [A Unified Framework for Data Poisoning Attack to Graph-based Semi-supervised Learning](#9)
    + [10] [A Restricted Black-box Adversarial Framework Towards Attacking Graph Embedding Models](#10)
    + [11] [Adversarial Examples on Graph Data: Deep Insights Into Attack and Defense](#11)
    + [12] [Multiscale Evolutionary Perturbation Attack on Community Detection](#12)
    + [13] [Adversarial Attacks on Graph Neural Networks via Meta Learning](#13)
    + [14] [Adversarial Attacks on Node Embeddings via Graph Poisoning](#14)
    + [15] [Time-aware Gradient Attack on Dynamic Network Link Prediction](#15)
    + [16] [Attacking Graph Convolutional Networks via Rewiring](#16)
    + [17] [Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective](#17)
    + [18] [Unsupervised Euclidean Distance Attack on Network Embedding](#18)
    + [19] [Generalizable Adversarial Attacks Using Generative Models](#19)
    + [20] [Data Poisoning Attack Against Knowledge Graph Embedding](#20)
    + [21] [Attacking Graph-Based Classification via Manipulating the Graph Structure](#21)
    + [22] [Ga Based Q-Attack on Community Detection](#22)
    + [23] [αcyber: Enhancing Robustness of Android Malware Detection System Against Adversarial Attacks on Heterogeneous Graph Based Model](#23)
    + [24] [Data Poisoning Attacks on Neighborhood-based Recommender Systems](#24)
    + [25] [Adversarial Recommendation: Attack of the Learned Fake Users](#25)
    + [26] [Attack Graph Convolutional Networks by Adding Fake Nodes](#26)
    + [27] [Attack Tolerance of Link Prediction Algorithms: How to Hide Your Relations in A Social Network](#27)
    + [28] [Link Prediction Adversarial Attack](#28)
    + [29] [Adversarial Attack on Graph Structured Data](#29)
    + [30] [Adversarial Attacks on Neural Networks for Graph Data](#30)
    + [31] [Fast Gradient Attack on Network Embedding](#31)
    + [32] [Data Poisoning Attack Against Unsupervised Node Embedding Methods](#32)
    + [33] [Attacking Similarity-Based Link Prediction in Social Networks](#33)
    + [34] [Practical Attacks Against Graph-Based Clustering](#34)

+ [Attack Type](#Type) (Incoming)

<a class="toc" id ="toc2"></a>

+ [Defense](#defense)
    - [1] [Transferring Robustness for Graph Neural Network Against Poisoning Attacks](#101)
    - [2] [Certified Robustness of Community Detection against Adversarial Structural Perturbation via Randomized Smoothing](#102)
    - [3] [Power Up! Robust Graph Convolutional Network Against Evasion Attacks Based on Graph Powering](#103)
    - [4] [How Robust Are Graph Neural Networks to Structural Noise?](#104)
    - [5] [All You Need is Low (Rank): Defending Against Adversarial Attacks on Graphs](#105)
    - [6] [Certifiable Robustness and Robust Training for Graph Convolutional Networks](#106)
    - [7] [Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective](#107)
    - [8] [Adversarial Examples on Graph Data: Deep Insights Into Attack and Defense](#108)
    - [9] [Adversarial Defense Framework for Graph Neural Network](#109)
    - [10] [Investigating Robustness and Interpretability of Link Prediction via Adversarial Modifications](#110)
    - [11] [Robust Graph Convolutional Networks Against Adversarial Attacks](#111)
    - [12] [Can Adversarial Network Attack Be Defended?](#112)
    - [13] [Virtual Adversarial Training on Graph Convolutional Networks in Node Classification](#113)
    - [14] [Comparing and Detecting Adversarial Attacks for Graph Deep Learning](#114)
    - [15] [Graph Adversarial Training: Dynamically Regularizing Based on Graph Structure](#115)
    - [16] [Characterizing Malicious Edges Targeting on Graph Neural Networks](#116)
    - [17] [Batch Virtual Adversarial Training for Graph Convolutional Networks](#117)
    - [18] [GraphDefense: Towards Robust Graph Convolutional Networks](#118)
    - [19] [αcyber: Enhancing Robustness of Android Malware Detection System against Adversarial Attacks on Heterogeneous Graph based Model](#119)
    - [20] [Edge Dithering for Robust Adaptive Graph Convolutional Networks](#120)
    - [21] [GraphSAC: Detecting Anomalies in Large-scale Graphs](#121)
    - [22] [Certifiable Robustness to Graph Perturbations](#122)
    - [23] [Adversarial Robustness of Similarity-Based Link Prediction](#123)
    - [24] [Improving Robustness to Attacks Against Vertex Classification](#124)
    - [25] [Adversarial Personalized Ranking for Recommendation](#125)
    
+ [Baselines](#baseline)

+ [Metric](#metric)

+ [Survey](#survey)

+ [Citation](#citation)
 

# Attack
|Venue|Title|Model|Algorithm|Target Task|Target Model|Baseline|Metric|Dataset|Code|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Arxiv<br>2020|<a class="toc" id ="1"></a>[[1]](https://arxiv.org/abs/2002.11320)<br>[🔙](#toc1)|MGA|Gradient-based GCN|Node Classification,<br>Community Detection|GCN, DeepWalk, <br> Node2vec, GraphGAN,<br>LPA, Louvain|GradArgmax,<br>RL-S2V,<br>Nettack,<br>FGA|ASR, AML|Cora, Citeseer, Polblogs, <br>Dolphin, PloBook|-|
|Arxiv<br>2020|<a class="toc" id ="2"></a>[[2]](https://arxiv.org/abs/2002.01249)<br>[🔙](#toc1)|RLR, DALR, DILR|Random, <br>Degree-based|Network Structure|Physical Criteria|-| △M (AML),<br> △L, △C ,△D|Generated simplex networks|-|
|Arxiv<br>2020|<a class="toc" id ="3"></a>[[3]](https://arxiv.org/abs/2002.04784)<br>[🔙](#toc1)|GUA|Anchors identified (based on GCN)|Node Classification|GCN, DeepWalk, <br> Node2Vec, GAT|Random, VCA, FGA|AML, ASR|Cora, Citeseer, Polblogs,|-|
|WWW<br>2020|<a class="toc" id ="4"></a>[[4]](https://arxiv.org/abs/2001.07933)<br>[🔙](#toc1)|CD-ATTACK|Graph generation based on GCN|Community Detection|GCN,<br>Node2vec + K-means,<br>ComE|DICE, MBA, RTA|Hiding performance measure M1 & M2|DBLP,<br>Finance|[Link](https://github.com/halimiqi/CD-ATTACK)|
|AAMAS<br>2020|<a class="toc" id ="5"></a>[[5]](https://arxiv.org/abs/1910.11529)<br>[🔙](#toc1)|FPTA|-|Node Similarity|Node Similarity Measures|Random, Greedy,<br> High Jaccard Similarity (HJ)||Barabasi-Albert (BA),<br>Erdos-Renyi (ER)|
|Arxiv<br>2019|<a class="toc" id ="6"></a>[[6]](https://arxiv.org/abs/1909.06543)<br>[🔙](#toc1)|||Node Classification|GCN|-|
|Arxiv<br>2019|<a class="toc" id ="7"></a>[[7]](https://arxiv.org/abs/1905.01776)<br>[🔙](#toc1)| ||Vertex Nomination|VN Scheme|-|
|Arxiv<br>2019|<a class="toc" id ="8"></a>[[8]](https://deepai.org/publication/the-general-black-box-attack-method-for-graph-neural-networks)<br>[🔙](#toc1)| |||||||-|
|NIPS<br>2019|<a class="toc" id ="9"></a>[[9]](https://arxiv.org/abs/1910.14147)<br>[🔙](#toc1)| ||||||||
|AAAI<br>2020|<a class="toc" id ="10"></a>[[10]](https://arxiv.org/abs/1908.01297)<br>[🔙](#toc1)|GF-Attack|Graph signal processing with graph filter|Node Classification|GCN, SGC,<br>DeepWalk,<br>LINE|Random,<br>Degree,<br>RL-S2V, <img src="http://latex.codecogs.com/gif.latex?A_{class}">|Accuracy|Cora,<br>CiteSeer,<br>Pubmed|[Link](https://github.com/SwiftieH/GFAttack)|
|Arxiv<br>2019|<a class="toc" id ="11"></a>[[11]](https://arxiv.org/abs/1910.09741)<br>[🔙](#toc1)|IG-FGSM,<br>IG-JSMA|Gradient-based GCN|Node Classification|GCN|FGSM,<br> JSMA,<br> Nettack|Classification Margin,<br>Accuracy|Cora,<br>CiteSeer,<br>PolBlogs|-|
|Arxiv<br>2019|<a class="toc" id ="12"></a>[[12]](https://arxiv.org/abs/1910.09741)<br>[🔙](#toc1)|EPA|Genetic algorithm based on|Community Detection|GRE, INF, LOU|<img src="http://latex.codecogs.com/gif.latex?A_Q, A_S">, <img src="http://latex.codecogs.com/gif.latex?A_B, A_D,"><br> <img src="http://latex.codecogs.com/gif.latex?D_S, D_W">|NMI, ARI|Synthetic networks,<br>Football,<br>Email,<br>Polblogs|-|
|ICLR<br>2019|<a class="toc" id ="13"></a>[[13]](https://arxiv.org/abs/1902.08412)<br>[🔙](#toc1)|Meta-Self<br>Meta-Train|Gradient-based GCN|Node Classification|GCN,<br>CLN,<br>DeepWalk|DICE, <br>Nettack, <br>First-order|Misclassification Rate, <br>Accuracy |Cora,<br>CiteSeer,<br>PolBlogs,<br>PubMed|[Link](https://github.com/danielzuegner/gnn-meta-attack)|
|ICML<br>2019|<a class="toc" id ="14"></a>[[14]](https://arxiv.org/abs/1809.01093)<br>[🔙](#toc1)|<img src="http://latex.codecogs.com/gif.latex?\mathcal{A}_{DW2}"><br><img src="http://latex.codecogs.com/gif.latex?\mathcal{A}_{DW3}">|Gradient- based random walk|Node Classification,<br>Link Prediction|DeepWalk|<img src="http://latex.codecogs.com/gif.latex?\mathcal{B}_{rnd}"><br><img src="http://latex.codecogs.com/gif.latex?\mathcal{B}_{eig}"><br><img src="http://latex.codecogs.com/gif.latex?\mathcal{B}_{deg}">|F1 Score,<br>Classification Margin<br>|Cora,<br>Citeseer,<br>PolBlogs|[Link](https://github.com/abojchevski/node_embedding_attack)
|Arxiv<br>2019|<a class="toc" id ="15"></a>[[15]](http://arxiv.org/abs/1911.10561)<br>[🔙](#toc1)|TGA-Tra,<br>TGA-Gre|Gradient-based DDNE|Link Prediction|DDNE, ctRBM,<br>GTRBM,<br>dynAERNN|Random DGA,<br>CNA|ASR, AML |RADOSLAW,<br>LKML,<br>FB-WOSN|-|
|Arxiv<br>2019|<a class="toc" id ="16"></a>[[16]](https://arxiv.org/abs/1906.03750)<br>[🔙](#toc1)|ReWatt|Reinforcement learning based on GCN|Graph Classification|GCN|RL-S2V,<br>RA|ASR |REDDIT-MULTI-12K,<br>REDDIT-MULTI-5K,<br>IMDB-MULTI|-|
|IJCAI<br>2019|<a class="toc" id ="17"></a>[[17]](https://arxiv.org/abs/1906.04214)<br>[🔙](#toc1)|PGD<br>Min-Max|Gradient-based GCN|Node Classification|GCN|DICE,<br>Meta-Self,<br>Greedy|Misclassification Rate |Cora,<br>Citeseer|[Link](https://github.com/KaidiXu/GCN_ADV_Train)|
|Arxiv<br>2019|<a class="toc" id ="18"></a>[[18]](https://arxiv.org/abs/1905.11015)<br>[🔙](#toc1)|EDA|Genetic algorithm based on DeepWalk|Node Classification,<br>Community Detection|HOPE,<br>LPA,<br>EM,<br>DeepWalk|Random<br>DICE,<br>RLS,<br>DBA|NMI,<br>Micro-F1,<br>Macro-F1|Karate,<br>Game,<br>Dolphin|-|
|Arxiv<br>2019|<a class="toc" id ="19"></a>[[19]](https://arxiv.org/abs/1905.10864)<br>[🔙](#toc1)|DAGAER|Generative model based on VGAE|Node Classification|GCN|Nettack|ASR |Cora<br>CiteSeer|-|
|IJCAI<br>2019|<a class="toc" id ="20"></a>[[20]](https://arxiv.org/abs/1904.12052)<br>[🔙](#toc1)|-|Knowledge embedding|Fact Plausibility Prediction|TransE,<br>TransR,<br>RESCAL|RA|MRR,<br>HR@K|FB15k,<br>WN18|-|
|CCS<br>2019|<a class="toc" id ="21"></a>[[21]](https://arxiv.org/abs/1903.00553)<br>[🔙](#toc1)|-|Based on LinLBP|Node Classification,<br>Evasion|LinLBP, JWP,<br>LBP, RW, LINE,<br>DeepWalk,<br>Node2vec,<br>GCN|Random<br>Nettack|FNR ,<br>FPR |Facebook,<br>Enron,<br>Epinions,<br>Twitter,<br>Google+|-|
|TCSS<br>2019|<a class="toc" id ="22"></a>[[22]](https://arxiv.org/abs/1811.00430)<br>[🔙](#toc1)|Q-Attack|Genetic algorithm|Community Detection|FN, Lou, SOA,<br>LPA, INF,<br>Node2vec+KM|Random<br>CDA,<br>DBA|Modularity Q,<br>NMI|Karate,<br>Dolphins,<br>Football,<br>Polbooks|-|
|CIKM<br>2019|<a class="toc" id ="23"></a>[[23]](https://dl_acm.xilesou.top/citation.cfm?id=3357875)<br>[🔙](#toc1)|HG-Attack|Label propagation algorithm<br>Adding fake nodes|Malware Detection|Orig-HGC|AN-Attack|TP, TN, FP, FN, F1, <br>Precision, Recall, Accuracy|Tencent Security Lab Dataset|-|
|Arxiv<br>2019|<a class="toc" id ="24"></a>[[24]](https://arxiv.org/abs/1912.04109)<br>[🔙](#toc1)|UNAttack|Gradient-based similarity method<br>Adding fake nodes|Recommendation|Memory-based CF,<br>BPRMF, NCF|-|Hit@K|Filmtrust,<br>Movielens,<br>Amazon|-|
|Arxiv<br>2018|<a class="toc" id ="25"></a>[[25]](https://arxiv.org/abs/1809.08336)<br>[🔙](#toc1)|-|Gradient-based GAN, MF<br>Adding fake nodes|Recommendation|MF|Random, Average, Popular, Co-visitation|Attack Difference,<br>TVD, JS, Est., <br>Rank Loss @K,<br>Adversarial loss|Movielens 100K,<br>Movielens 1M|-|
|Arxiv<br>2018|<a class="toc" id ="26"></a>[[26]](https://arxiv.org/abs/1810.10751)<br>[🔙](#toc1)|Greedy<br>Greedy GAN|Gradient-based GCN, GAN|Node Classification|GCN|RA|Accuracy, <br>F1 Score, ASR |Cora, <br>Citeseer|-
|Arxiv<br>2018|<a class="toc" id ="27"></a>[[27]](https://arxiv.org/abs/1809.00152)<br>[🔙](#toc1)|CTR<br>OTC|Neighbour score based on graph structure|Link Prediction|Traditional Link Prediction Algs|-|AUC, AP|WTC 9/11,<br>ScaleFree,<br>Facebook,<br>Random network|-|
|Arxiv<br>2018|<a class="toc" id ="28"></a>[[28]](https://arxiv.org/abs/1810.01110)<br>[🔙](#toc1)|IGA|Gradient-based  GAE|Link Prediction|GAE, LRW <br>DeepWalk, <br>Node2vec, <br>CN, RA, Katz|RAN, <br>DICE, <br>GA|ASR , <br>AML|NS, <br>Yeast, <br>FaceBook|-|
|ICML<br>2018|<a class="toc" id ="29"></a>[[29]](https://arxiv.org/abs/1806.02371)<br>[🔙](#toc1)|RL-S2V|Reinforcement learning|Node/Graph Classification|GCN,<br>GNN|Random Sampling|Accuracy|Citeseer, <br>Cora, <br>Pubmed, <br>Finance|[Link](https://github.com/Hanjun-Dai/graph_adversarial_attack)
|KDD<br>2018|<a class="toc" id ="30"></a>[[30]](https://arxiv.org/abs/1805.07984)<br>[🔙](#toc1)|Nettack|Greedy search & gradient <br>based on GCN|Node Classification|GCN,<br>CLN,<br>DeepWalk|Rnd,<br>FGSM|Classification Margin,<br>Accuracy|Cora-ML,<br>Citeseer,<br>PolBlogs|[Link](https://github.com/danielzuegner/nettack)|
|Arxiv<br>2018|<a class="toc" id ="31"></a>[[31]](https://arxiv.org/abs/1809.02797)<br>[🔙](#toc1)|FGA|Gradient-based GCN|Node Classification,<br>Community Detection|GCN,<br>GraRep,<br>DeepWalk,<br>Node2vec,<br>LINE,<br>GraphGAN|Random<br>DICE,<br>Nettack|ASR, AML|Cora,<br>Citeseer,<br>PolBlogs|-|
|Arxiv<br>2018|<a class="toc" id ="32"></a>[[32]](https://arxiv.org/abs/1810.12881)<br>[🔙](#toc1)|Opt-attack|Gradient based on DeepWalk, LINE|Link Prediction|DeepWalk<br>LINE<br>Node2vec<br>SC<br>GAE|Random<br>PageRank,<br>Degree sum,,<br>Shortest path|Similarity Score <br>AP|Facebook,<br>Cora,<BR>Citeseer|-|
|AAMAS<br>2018|<a class="toc" id ="33"></a>[[33]](https://arxiv.org/abs/1809.08368)<br>[🔙](#toc1)|Approx-Local|Similarity methods|Link Prediction|Local&Global similarity metrics|RandomDel,<br>GreedyBase|Katz Similarity,<br>ACT Distance,<br>Similarity Score|Random network,<br>Facebook|-|
|CCS<br>2017|<a class="toc" id ="34"></a>[[34]](https://arxiv.org/abs/1708.09056)<br>[🔙](#toc1)|Targeted noise injection,<br>Small community attack|Noise injection|Graph Clustering,<br>Community Detection|SVD,<br>Node2vec,<br>Community Detection Algs|-|ASR, FPR |Reverse Engineered DGA Domains,<br>NXDOMAIN|-|

<!-- <a class="toc" id ="Type"></a>
# Attack Type
|Method|Targeted|Non-targeted|Black-box|Gray-box|Poisoning|Evasion|Topology|Feature|
|:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|[1] [A Restricted Black-box Adversarial Framework Towards Attacking Graph Embedding Models](#1)|
|[2] [Adversarial Examples on Graph Data: Deep Insights Into Attack and Defense](#2)|⭕|❌|❌|⭕|⭕|❌|⭕|⭕|
|[3] [Multiscale Evolutionary Perturbation Attack on Community Detection](#3)|
|[4] [Adversarial Attacks on Graph Neural Networks via Meta Learning](#4)|❌|⭕|⭕|⭕|⭕|❌|⭕|❌|
|[5] [Adversarial Attacks on Node Embeddings via Graph Poisoning](#5)|⭕|⭕|⭕|⭕|⭕|❌|⭕|❌|
|[6] [Time-aware Gradient Attack on Dynamic Network Link Prediction](#6)
|[7] [Attacking Graph Convolutional Networks via Rewiring](#7)|❌|⭕|⭕|⭕|❌|⭕|⭕|❌|
|[8] [Topology Attack and Defense for Graph Neural Networks: An Optimization rspective](#8)|❌|⭕|⭕|⭕|⭕|❌|⭕|❌|
|[9] [Unsupervised Euclidean Distance Attack on Network Embedding](#9)|❌|⭕|⭕|⭕|⭕|❌|⭕|❌
|[10] [Generalizable Adversarial Attacks Using Generative Models](#10)|⭕|❌|❌|⭕|❌|⭕|❌|⭕
|[11] [Data Poisoning Attack Against Knowledge Graph Embedding](#11)|⭕|❌|⭕|❌|⭕|❌|⭕|❌|
|[12] [Attacking Graph-Based Classification via Manipulating the Graph Structure](#12)|⭕|❌|⭕|⭕|⭕|❌|⭕|❌|
|[13] [Ga Based Q-Attack on Community Detection](#13)|
|[14] [αcyber: Enhancing Robustness of Android Malware Detection System Against Adversarial Attacks on Heterogeneous Graph Based Model](#14)|
|[15] [Data Poisoning Attacks on Neighborhood-based Recommender Systems](#15)|
|[16] [Adversarial Recommendation: Attack of the Learned Fake Users](#16)|
|[17] [Attack Graph Convolutional Networks by Adding Fake Nodes](#17)|⭕|⭕|❌|⭕|⭕|❌|⭕|⭕|
|[18] [Attack Tolerance of Link Prediction Algorithms: How to Hide Your Relations in Social Network](#18)|⭕|❌|⭕|❌|❌|⭕|⭕|❌|
|[19] [Link Prediction Adversarial Attack](#19)|⭕|❌|⭕|⭕|❌|⭕|⭕|❌|
|[20] [Adversarial Attack on Graph Structured Data](#20)|⭕|❌|⭕|⭕|❌|⭕|⭕|❌|
|[21] [Adversarial Attacks on Neural Networks for Graph Data](#21)|⭕|❌|⭕|⭕|⭕|❌|⭕|⭕|
|[22] [Fast Gradient Attack on Network Embedding](#22)|⭕|❌|⭕|⭕|⭕|❌|⭕|❌|
|[23] [Data Poisoning Attack Against Unsupervised Node Embedding Methods](#23)|⭕|⭕|⭕|⭕|⭕|❌|⭕|❌|
|[24] [Attacking Similarity-Based Link Prediction in Social Networks](#24)|
|[25] [Practical Attacks Against Graph-Based Clustering](#25)| -->

![Attack](imgs/Attack.png)
![Targeted attack](imgs/attack_demo.png){:height="50%" width="50%"}
![Poisoning attack and Evasion attack](imgs/p_and_e.png)

 

# Defense

| Venue| Title | Model | Algorithm | Defense Type | Target Task | Target Model | Baseline | Metric | Dataset | Code |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| WSDM 2020 | <a class="toc" id ="101"></a> [[1]](https://arxiv.org/abs/1908.07558) [🔙](#toc2) | PA-GNN | Penalized Aggregation, Meta Learning | Structure Based | Node Classification | GNN | GCN, GAT, PreProcess, RGCN, VPN | Accuracy | Pubmed, Reddit, Yelp-Small, Yelp-Large | - |
| WWW 2020 | <a class="toc" id ="102"></a> [[2]](https://arxiv.org/abs/2002.03421) [🔙](#toc2) ||| |||||||| - |
| ICLR 2020 OpenReview | <a class="toc" id ="103"></a> [[3]](https://arxiv.org/abs/1905.10029) [🔙](#toc2) | r-GCN, VPN | Graph Powering | Objective Based | Node Classification | GCN | ManiReg, SemiEmb, LP, DeepWalk, ICA, Planetoid, Vanilla GCN | Accuracy, Robustness Merit,<br>Attack Deterioration | Citeseer, Cora, Pubmed | - |
| Arxiv 2019 | <a class="toc" id ="104"></a> [[4]](https://arxiv.org/abs/1912.10206) [🔙](#toc2) ||| |||||||| - |
| WSDM 2019 | <a class="toc" id ="105"></a> [[5]](https://dl.acm.org/doi/abs/10.1145/3336191.3371789) [🔙](#toc2) ||| |||||||| [Link](https://github.com/DSE-MSU/DeepRobust) |
| KDD 2019 | <a class="toc" id ="106"></a> [[6]](https://arxiv.org/abs/1906.12269) [🔙](#toc2) | GNN (trained with RH-U) | Robustness Certification, Objective Based | Hybrid | Node Classification | GNN, GCN | GNN (trained with CE, RCE, RH) | Accuracy, Averaged Worst-case Margin | Citeseer, Cora-ML, Pubmed | [Link](https://www.kdd.in.tum.de/research/robust-gcn/) |
| IJCAI 2019 | <a class="toc" id ="107"></a> [[7]](https://arxiv.org/abs/1906.04214) [🔙](#toc2) | - | Adversarial Training | Adversarial Training | Node Classification | GCN | GCN | Misclassification Rate <br>Accuracy | Citeseer, Cora | [Link](https://github.com/KaidiXu/GCN_ADV_Train) |
| IJCAI 2019 | <a class="toc" id ="108"></a> [[8]](https://arxiv.org/abs/1903.01610) [🔙](#toc2) | - | Drop Edges | Preprocessing | Node Classification | GCN | GCN | Classfication Margin, Accuracy | Cora-ML, Citeseer, PolBlogs | [Link](https://github.com/stellargraph/stellargraph/) |
| Arxiv 2019 | <a class="toc" id ="109"></a> [[9]](https://arxiv.org/abs/1905.03679) [🔙](#toc2) | DefNet | GAN,<br>GER,<br>ACL | Hybrid | Node Classification | GCN, GraphSAGE | GCN, GraphSage | Classfication Margin | Cora, Citeseer, PolBlogs | - |
| NAACL 2019 | <a class="toc" id ="110"></a> [[10]](https://arxiv.org/abs/1905.00563) [🔙](#toc2) | CRIAGE | Adversarial Modification | Robustness Evaluation | Link Prediction | Knowledge Graph Embedding | - | Hits@K, MRR | Nations, Kinship, WN18, YAGO3-10 | - |
| KDD 2019 | <a class="toc" id ="111"></a> [[11]](http://pengcui.thumedialab.com/papers/RGCN.pdf) [🔙](#toc2) | RGCN | Gaussian-based Graph Convolution | Structure Based | Node Classification | GCN | GCN, GAT | Accuracy | Cora, Citeseer, Pubmed | [Link](https://github.com/thumanlab/nrlweb/blob/master/static/assets/download/RGCN.zip) |
| Arxiv 2019 | <a class="toc" id ="112"></a> [[12]](https://arxiv.org/abs/1903.05994) [🔙](#toc2) | Global-AT, Target-AT, SD, SCEL | Adversarial Training, Smooth Defense | Hybrid | Node Classification | GNN | AT | ADR, ACD | PolBlogs, Cora, Citeseer | - |
| PRCV 2019 | <a class="toc" id ="113"></a> [[13]](https://arxiv.org/abs/1902.11045) [🔙](#toc2) | SVAT, DVAT | Virtual Adversarial Training | Adversarial Training | Node Classification | GCN | GCN | Accuracy | Cora, Citeseer, Pubmed | - |
| RLGM@ICLR 2019 | <a class="toc" id ="114"></a> [[14]](https://rlgm.github.io/papers/57.pdf) [🔙](#toc2) | - | KL Divergence | Detection Based | Node Classification | GCN, GAT | - | Classfication Margin, Accuracy,<br>ROC, AUC | Cora, Citeseer, PolBlogs | - |
| Arxiv 2019 | <a class="toc" id ="115"></a> [[15]](https://arxiv.org/abs/1902.08226) [🔙](#toc2) | GCN-GATV | Graph Adversarial Training, Virtual Adversarial Training | Adversarial Training | Node Classification | GCN | LP, DeepWalk, SemiEmb, Planetoid, GCN, GraphSGAN | Accuracy | Citeseer, Cora, NELL | - |
| ICLR 2019 OpenReview | <a class="toc" id ="116"></a> [[16]](https://openreview.net/forum?id=HJxdAoCcYX) [🔙](#toc2) | SL, OD, GGD, LP+GGD, ENS | Link Prediction, Subsampling, Neighbour Analysis| Hybrid | Node Classification | GNN, GCN | LP | AUC | Cora, Citeseer | - |
| ICML 2019 | <a class="toc" id ="117"></a> [[17]](https://arxiv.org/abs/1902.09192) [🔙](#toc2) | S-BVAT, O-BVAT | Batch Virtual Adversarial Training | Adversarial Training | Node Classification | GCN | ManiReg, SemiEmb, LP, DeepWalk, Planetoid, Monet, GAT, GPNN, GCN, VAT | Accuracy | Cora, Citeseer, Pubmed, Nell | [Link](https://github.com/thudzj/BVAT) |
| Arxiv 2019 | <a class="toc" id ="118"></a> [[18]](https://arxiv.org/abs/1911.04429) [🔙](#toc2) | GraphDefense | Adversarial Training | Adversarial Training | Node Classification | GCN | Drop Edges, Discrete Adversarial Training | Accuracy | Cora, Citeseer, Reddit | - |
| CIKM 2019 | <a class="toc" id ="119"></a> [[19]](https://dl.acm.org/citation.cfm?id=3357875) [🔙](#toc2) | Rad-HGC | HG-Defense | Detection Based | Malware Detection | Malware Detection System | FakeBank, CryptoMiner, AppCracked, MalFlayer, GameTrojan, BlackBaby, SDKSmartPush, ... | Detection Rate | Tencent Security Lab Dataset | - |
| Arxiv 2019 | <a class="toc" id ="120"></a> [[20]](https://arxiv.org/abs/1910.09590) [🔙](#toc2) | AGCN | Adaptive GCN with Edge Dithering | Structure Based  | Node Classification | GCN | GCN | Accuracy | Citeseer, PolBlogs, Cora, Pubmed | - |
| Arxiv 2019 | <a class="toc" id ="121"></a> [[21]](https://arxiv.org/abs/1910.09589) [🔙](#toc2) | GraphSVC | Random Sampling, Consensus | Detection Based | Anomaly Detection | Anomaly Model | GAE, Amen, Radar, Degree, Cut ratio, Flake, Conductance | AUC | Citeseer, PolBlogs, Cora, Pubmed | - |
| NIPS 2019 | <a class="toc" id ="122"></a> [[22]](http://papers.nips.cc/paper/9041-certifiable-robustness-to-graph-perturbations) [🔙](#toc2) | GNN (train with <img src="http://latex.codecogs.com/gif.latex?L_{RCE}">, <img src="http://latex.codecogs.com/gif.latex?L_{CEM}"> ) | Robustness Certification, Objective Based | Hybrid | Node Classification | GNN | GNN | Accuracy, Worst-case Margin | Cora-ML, Citeseer, Pubmed | [link](https://github.com/abojchevski/graph_cert) |
| ICDM 2019 | <a class="toc" id ="123"></a> [[23]](https://arxiv.org/abs/1909.01432) [🔙](#toc2) | IDOpt, IDRank | Integer Program, Edge Ranking | Heuristic Algorithm | Link Prediction | Similarity-based Link Prediction Models | PPN | DPR | PA, PLD, TVShow, Gov | - |
| MLG@KDD 2019 | <a class="toc" id ="124"></a> [[24]](http://eliassi.org/papers/benmiller-mlg2019.pdf) [🔙](#toc2) | SVM with a radial basis function kernel | Augmented Feature, Edge Selecting | Hybrid | Node Classification | SVM | GCN | Classification Marigin | Cora, Citeseer | - |
| SIGIR 2018 | <a class="toc" id ="125"></a> [[25]](https://dl.acm.org/citation.cfm?id=3209981) [🔙](#toc2) | APR, AMF | Adversarial Training based on MF-BPR | Adversarial Training | Recommendation | MF-BPR | ItemPop, MF-BPR, CDAE, NeuMF, IRGAN | HR, NDCG | Yelp, Pinterest, Gowalla | [Link](https://github.com/hexiangnan/adversarial_personalized_ranking) |

<a class="toc" id ="baseline"></a>
# Baselines
| Baseline | Venue | Paper | Code |
|:-:|:-:|:-:|:-:|
| DICE | Nature Human Behaviour 2018 | [Hiding Individuals and Communities in a Social Network](https://arxiv.org/abs/1608.00375) | [Link](https://github.com/JHL-HUST/HiCode) |
| Nettack | KDD 2018 | [Adversarial Attacks on Neural Networks for Graph Data](https://arxiv.org/abs/1805.07984) | [Link](https://github.com/danielzuegner/nettack) |
| First-order | ICML 2017 | [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400) | [Link](https://github.com/cbfinn/maml) |
| RL-S2V | ICML 2018 | [Adversarial Attack on Graph Structured Data](https://arxiv.org/abs/1806.02371) | [Link](https://github.com/Hanjun-Dai/graph_adversarial_attack) |
| Meta-Self | ICLR 2019 | [Adversarial Attacks on Graph Neural Networks via Meta Learning](https://arxiv.org/abs/1902.08412) | [Link](https://github.com/danielzuegner/gnn-meta-attack) |
| Greedy | ICLR 2019 | [Adversarial Attacks on Graph Neural Networks via Meta Learning](https://arxiv.org/abs/1902.08412) | [Link](https://github.com/danielzuegner/gnn-meta-attack) |
| DBA | IEEE Transactions 2019 | [GA Based Q-Attack on Community Detection](https://arxiv.org/abs/1811.00430) | - |
| CDA | IEEE Transactions 2019 | [GA Based Q-Attack on Community Detection](https://arxiv.org/abs/1811.00430) | - |
| GA (Gradient based)|  ECML PKDD 2013 | [Evasion Attacks Against Machine Learning at Test Time](https://arxiv.org/abs/1708.06131) | [Link](https://github.com/bethgelab/foolbox/blob/master/foolbox/attacks/gradient.py) |
| FGSM | ICLR 2015 | [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) | [Link](https://github.com/1Konny/FGSM) |
| PageRank | VLDB 2010 | [Fast Incremental and Personalized PageRank](https://arxiv.org/abs/1006.2880) | [Link](https://github.com/alexander-stannat/Incremental-Pagerank) |
| GNN | IEEE Transactions 2009 | [The Graph Neural Network Model](https://arxiv.org/abs/1412.6572v3) | [Link](https://github.com/SeongokRyu/Graph-neural-networks) |
| GCN | ICLR 2017 | [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) | [Link](https://github.com/tkipf/gcn) |
| ManiReg | JMLR 2006 | [Manifold Regularization: A Geometric Framework for Learning from Labeled and Unlabeled Examples](http://www.jmlr.org/papers/v7/belkin06a.html) | [Link](https://github.com/snehchav/Semi-Supervised-Image-Classification) |
| SemiEmb | ICML 2008 | [Deep Learning via Semi-supervised Embedding](https://link.springer.com/chapter/10.1007/978-3-642-35289-8_34) | [Link](https://github.com/yangminz/Semi-supervised_Embedding) |
| LP | ICML 2003 | [Semi-Supervised Learning Using Gaussian Fields and Harmonic Functions](https://www.semanticscholar.org/paper/Semi-Supervised-Learning-Using-Gaussian-Fields-and-Zhu-Ghahramani/02485a373142312c354b79552b3d326913eaf86d) | [Link](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/semi_supervised/label_propagation.py) |
| Deepwalk | KDD 2014 | [DeepWalk: Online Learning of Social Representations](https://arxiv.org/abs/1403.6652) | [Link](https://github.com/phanein/deepwalk) |
| ICA | ICML 2003 | [Link-based classification](https://link.springer.com/chapter/10.1007/1-84628-284-5_7) | [Link](https://github.com/tkipf/ica) |
| Planetoid | ICML 2016 | [Revisiting Semi-Supervised Learning with Graph Embeddings](https://arxiv.org/abs/1603.08861) | [Link](https://github.com/kimiyoung/planetoid) |
| GraphSage | NIPS 2017 | [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) | [Link](https://github.com/williamleif/GraphSAGE) |
| DistMult | ICLR 2015 | [Embedding Entities and Relations for Learning and Inference in Knowledge Bases](https://arxiv.org/abs/1412.6575) | [Link](https://github.com/mana-ysh/knowledge-graph-embeddings) |
| ConvE | AAAI 2018 | [Convolutional 2D Knowledge Graph Embeddings](https://arxiv.org/abs/1707.01476) | [Link](https://github.com/TimDettmers/ConvE) |
| GAT | ICLR 2018 | [Graph Attention Networks](https://arxiv.org/abs/1710.10903) | [Link](https://github.com/PetarV-/GAT) |
| AT | ICLR 2015 | [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) | [Link](https://github.com/tensorflow/cleverhans) |
| BGCN | AAAI 2019 | [Bayesian graph convolutional neural networks for semi-supervised classification](https://arxiv.org/abs/1811.11103) | - |
| GraphSGAN | ACM 2018 | [Semi-supervised Learning on Graphs with Generative Adversarial Nets](https://arxiv.org/abs/1809.00130) | [Link](https://github.com/THUDM/GraphSGAN) |
| Monet | CVPR 2017 | [Geometric deep learning on graphs and manifolds using mixture model CNNs](https://arxiv.org/abs/1611.08402) | [Link](https://github.com/pierrebaque/GeometricConvolutionsBench) |
| GPNN | CVPR 2018 | [Graph Partition Neural Networks for Semi-Supervised Classification](https://arxiv.org/abs/1803.06272) | [Link](https://github.com/microsoft/graph-partition-neural-network-samples) |
| VAT | IEEE Transactions 2018 | [Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning](https://arxiv.org/abs/1704.03976) | [Link](https://github.com/takerum/vat_tf) |


<a class="toc" id ="metric"></a>

# Metric
[🔙](#toc1)
![Metric](imgs/Metric.png)

<a class="toc" id ="Survey"></a>

# Survey
[🔙](#toc1)
[A Survey of Adversarial Learning on Graphs](https://arxiv.org/abs/2003.05730)

<a class="toc" id ="Citation"></a>

# Citation
[🔙](#toc1)
If you find this repo useful, please cite:
```
@misc{chen2020survey,
    title={A Survey of Adversarial Learning on Graphs},
    author={Liang Chen and Jintang Li and Jiaying Peng and Tao Xie and Zengxu Cao and Kun Xu and Xiangnan He and Zibin Zheng},
    year={2020},
    eprint={2003.05730},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
