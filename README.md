# Table of Contents

<a class="toc" id ="toc1"></a>

+ [Attack](#attack)
    + [1] [Adversarial Attacks On Graph Neural Networks Via Meta Learning](#1)
    + [2] [Adversarial Attacks On Node Embeddings Via Graph Poisoning](#2)
    + [3] [Attacking Graph Convolutional Networks Via Rewiring](#3)
    + [4] [Topology Attack And Defense For Graph Neural Networks: An Optimization Perspective](#4)
    + [5] [Unsupervised Euclidean Distance Attack On Network Embedding](#5)
    + [6] [Generalizable Adversarial Attacks Using Generative Models](#6)
    + [7] [Data Poisoning Attack Against Knowledge Graph Embedding](#7)
    + [8] [Attacking Graph-Based Classification Via Manipulating The Graph Structure](#8)
    + [9] [Ga Based Q-Attack On Community Detection](#9)
    + [10] [$\alpha$cyber: Enhancing robustness of android malware detection system against adversarial attacks on heterogeneous graph based model](#10)
    + [11] [Data Poisoning Attacks on Neighborhood-based Recommender Systems](#11)
    + [12] [Adversarial Recommendation: Attack of the Learned Fake Users](#12)
    + [13] [Attack Graph Convolutional Networks By Adding Fake Nodes](#13)
    + [14] [Attack Tolerance Of Link Prediction Algorithms: How To Hide Your Relations In A Social Network](#14)
    + [15] [Link Prediction Adversarial Attack](#15)
    + [16] [Adversarial Attack On Graph Structured Data](#16)
    + [17] [Adversarial Attacks On Neural Networks For Graph Data](#17)
    + [18] [Fast Gradient Attack On Network Embedding](#18)
    + [19] [Data Poisoning Attack Against Unsupervised Node Embedding Methods](#19)
    + [20] [Attacking Similarity-Based Link Prediction In Social Networks](#20)
    + [21] [Practical Attacks Against Graph-Based Clustering](#21)

+ [Attack Type](#Type)

<a class="toc" id ="toc2"></a>

- [Defense](#defense)
    - [1] [Certifiable Robustness And Robust Training For Graph Convolutional Networks](#101)
    - [2] [Topology Attack And Defense For Graph Neural Networks: An Optimization Perspective](#102)
    - [3] [Power Up! Robust Graph Convolutional Network Against Evasion Attacks Based On Graph Powering](#103)
    - [4] [Adversarial Examples On Graph Data: Deep Insights Into Attack And Defense](#104)
    - [5] [Adversarial Defense Framework For Graph Neural Network](#105)
    - [6] [Investigating Robustness And Interpretability Of Link Prediction Via Adversarial Modifications](#106)
    - [7] [Robust Graph Convolutional Networks Against Adversarial Attacks](#107)
    - [8] [Can Adversarial Network Attack Be Defended?](#108)
    - [9] [Virtual Adversarial Training On Graph Convolutional Networks In Node Classification](#109)
    - [10] [Comparing And Detecting Adversarial Attacks For Graph Deep Learning](#110)
    - [11] [Graph Adversarial Training: Dynamically Regularizing Based On Graph Structure](#111)
    - [12] [Characterizing Malicious Edges Targeting On Graph Neural Networks](#112)
    - [13] [Batch Virtual Adversarial Training For Graph Convolutional Networks](#113)
    - [14] [Robust Graph Neural Network Against Poisoning Atacks via Transfer Learning](#114)
    - [15] [GraphDefense: Towards Robust Graph Convolutional Networks](#115)
    - [16] [Adversarial Personalized Ranking for Recommendation](#116)
    - [17] [αcyber: Enhancing Robustness of Android Malware Detection System against Adversarial Attacks on Heterogeneous Graph based Model](#117)
    
- [Baselines](#Baselines)

- [Metric](#Metric)

# Attack
|Venue|Title|Model|Algorithm|Target Task|Target Model|Baseline|Metric\*|Dataset|Code|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|ICLR<br>2019|<a class="toc" id ="1"></a>[[1]](https://arxiv.org/abs/1902.08412)<br>[🔙](#toc1)|Meta-Self<br>Meta-Train|Gradient-based GCN|Node Classification|GCN,<br>CLN,<br>DeepWalk|DICE, <br>NETTACK, <br>First-order|Misclassification Rate, <br>Accuracy |Cora,<br>CiteSeer,<br>PolBlogs,<br>PubMed|[Link](https://www.kdd.in.tum.de/research/gnn-meta-attack/)|
|ICML<br>2019|<a class="toc" id ="2"></a>[[2]](https://arxiv.org/abs/1809.01093)<br>[🔙](#toc1)|<img src="http://latex.codecogs.com/gif.latex?\mathcal{A}_{DW2}"><br><img src="http://latex.codecogs.com/gif.latex?\mathcal{A}_{DW3}">|Gradient- based random walk|Node Classification,<br>Link Prediction|DeepWalk|<img src="http://latex.codecogs.com/gif.latex?\mathcal{B}_{rnd}"><br><img src="http://latex.codecogs.com/gif.latex?\mathcal{B}_{eig}"><br><img src="http://latex.codecogs.com/gif.latex?\mathcal{B}_{deg}">|F1 Score,<br>Classification Margin<br>|Cora,<br>Citeseer,<br>PolBlogs|[Link](https://www.kdd.in.tum.de/research/node_embedding_attack/)
|Arxiv<br>2019|<a class="toc" id ="3"></a>[[3]](https://arxiv.org/abs/1906.03750)<br>[🔙](#toc1)|ReWatt|Reinforcement learning based on GCN|Graph Classification|GCN|RL-S2V,<br>RA|ASR |REDDIT-MULTI-12K,<br>REDDIT-MULTI-5K,<br>IMDB-MULTI|-|
|IJCAI<br>2019|<a class="toc" id ="4"></a>[[4]](https://arxiv.org/abs/1906.04214)<br>[🔙](#toc1)|PGD<br>Min-Max|Gradient-based GCN|Node Classification|GCN|DICE,<br>Meta-Self,<br>Greedy|Misclassification Rate |Cora,<br>Citeseer|[Link](https://github.com/KaidiXu/GCN_ADV_Train)|
|Arxiv<br>2019|<a class="toc" id ="5"></a>[[5]](https://arxiv.org/abs/1905.11015)<br>[🔙](#toc1)|EDA|Genetic algorithm based on DeepWalk|Node Classification,<br>Community Detection|HOPE,<br>LPA,<br>EM,<br>DeepWalk|RA,<br>DICE,<br>RLS,<br>DBA|NMI,<br>Micro-F1,<br>Macro-F1|Karate,<br>Game,<br>Dolphin|-|
|Arxiv<br>2019|<a class="toc" id ="6"></a>[[6]](https://arxiv.org/abs/1905.10864)<br>[🔙](#toc1)|DAGAER|Generative model based on VGAE|Node Classification|GCN|NETTACK|ASR |Cora<br>CiteSeer|-|
|Arxiv<br>2019|<a class="toc" id ="7"></a>[[7]](https://arxiv.org/abs/1904.12052)<br>[🔙](#toc1)|-|Knowledge embedding|Fact Plausibility Prediction|TransE,<br>TransR,<br>RESCAL|RA|MRR,<br>HR@K|FB15k,<br>WN18|-|
|Arxiv<br>2019|<a class="toc" id ="8"></a>[[8]](https://arxiv.org/abs/1903.00553)<br>[🔙](#toc1)|-|Based on LinLBP|Node Classification,<br>Evasion|LinLBP, JWP,<br>LBP, RW, LINE,<br>DeepWalk,<br>Node2vec,<br>GCN|RA,<br>NETTACK|FNR ,<br>FPR |Facebook,<br>Enron,<br>Epinions,<br>Twitter,<br>Google+|-|
|TCSS<br>2019|<a class="toc" id ="9"></a>[[9]](https://arxiv.org/abs/1811.00430)<br>[🔙](#toc1)|Q-Attack|Genetic algorithm|Community Detection|FN, Lou, SOA,<br>LPA, INF,<br>Node2vec+KM|RA,<br>CDA,<br>DBA|Modularity Q,<br>NMI|Karate,<br>Dolphins,<br>Football,<br>Polbooks|-|
|CIKM<br>2019|<a class="toc" id ="10"></a>[[10]](https://dl_acm.xilesou.top/citation.cfm?id=3357875)<br>[🔙](#toc1)|HG-Attack|Label propagation algorithm<br>Adding fake nodes|Malware Detection|Orig-HGC|AN-Attack|TP, TN, FP, FN, F1, <br>Precision, Recall, Accuracy|Tencent Security Lab Dataset|-|
|Arxiv<br>2019|<a class="toc" id ="11"></a>[[11]](https://arxiv.org/abs/1912.04109)<br>[🔙](#toc1)|UNAttack|Gradient-based similarity method<br>Adding fake nodes|Recommendation|Memory-based CF,<br>BPRMF, NCF|-|Hit@K|Filmtrust,<br>Movielens,<br>Amazon|-|
|Arxiv<br>2018|<a class="toc" id ="12"></a>[[12]](https://arxiv.org/abs/1809.08336)<br>[🔙](#toc1)|-|Gradient-based GAN, MF<br>Adding fake nodes|Recommendation|MF|Random, Average, Popular, Co-visitation|Attack Difference,<br>TVD, JS, Est., <br>Rank Loss @K,<br>Adversarial loss|Movielens 100K,<br>Movielens 1M|-|
|Arxiv<br>2018|<a class="toc" id ="13"></a>[[13]](https://arxiv.org/abs/1810.10751)<br>[🔙](#toc1)|Greedy<br>Greedy GAN|Gradient-based GCN, GAN|Node Classification|GCN|RA|Accuracy, <br>F1 Score, ASR |Cora, <br>Citeseer|-
|Arxiv<br>2018|<a class="toc" id ="14"></a>[[14]](https://arxiv.org/abs/1809.00152)<br>[🔙](#toc1)|CTR<br>OTC|Neighbour score based on graph structure|Link Prediction|Traditional Link Prediction Algs|-|AUC, AP|WTC 9/11,<br>ScaleFree,<br>Facebook,<br>Random network|-|
|Arxiv<br>2018|<a class="toc" id ="15"></a>[[15]](https://arxiv.org/abs/1810.01110)<br>[🔙](#toc1)|IGA|Gradient-based  GAE|Link Prediction|GAE, LRW <br>DeepWalk, <br>Node2vec, <br>CN, RA, Katz|RAN, <br>DICE, <br>GA|ASR , <br>AML|NS, <br>Yeast, <br>FaceBook|-|
|ICML<br>2018|<a class="toc" id ="16"></a>[[16]](https://arxiv.org/abs/1806.02371)<br>[🔙](#toc1)|RL-S2V|Reinforcement learning|Node/Graph Classification|GCN,<br>GNN|Random Sampling|Accuracy|Citeseer, <br>Cora, <br>Pubmed, <br>Finance|[Link](https://github.com/Hanjun-Dai/graph_adversarial_attack)
|KDD<br>2018|<a class="toc" id ="17"></a>[[17]](https://arxiv.org/abs/1805.07984)<br>[🔙](#toc1)|Nettack|Fast computation <br>based on GCN|Node Classification|GCN,<br>CLN,<br>DeepWalk|Rnd,<br>FGSM|Classification Margin,<br>Accuracy|Cora-ML,<br>Citeseer,<br>PolBlogs|[Link](https://github.com/danielzuegner/nettack)|
|Arxiv<br>2018|<a class="toc" id ="18"></a>[[18]](https://arxiv.org/abs/1809.02797)<br>[🔙](#toc1)|FGA|Gradient-based GCN|Node Classification,<br>Community Detection|GCN,<br>GraRep,<br>DeepWalk,<br>Node2vec,<br>LINE,<br>GraphGAN|RA,<br>DICE,<br>NETTACK|ASR, AML|Cora,<br>Citeseer,<br>PolBlogs|-|
|Arxiv<br>2018|<a class="toc" id ="19"></a>[[19]](https://arxiv.org/abs/1810.12881)<br>[🔙](#toc1)|Opt-attack|Gradient based on DeepWalk, LINE|Link Prediction|DeepWalk<br>LINE<br>Node2vec<br>SC<br>GAE|RA<br>PageRank<br>Degree sum<br>Shortest path|Similarity Score <br>AP|Facebook<br>Cora<BR>Citeseer|-|
|Arxiv<br>2018|<a class="toc" id ="20"></a>[[20]](https://arxiv.org/abs/1809.08368)<br>[🔙](#toc1)|Approx-Local|Similarity methods|Link Prediction|Local&Global similarity metrics|RandomDel,<br>GreedyBase|Katz Similarity,<br>ACT Distance,<br>Similarity Score|Random network,<br>Facebook|-|
|CCS<br>2017|<a class="toc" id ="21"></a>[[21]](https://arxiv.org/abs/1708.09056)<br>[🔙](#toc1)|Targeted noise injection,<br>Small community attack|Noise injection|Graph Clustering,<br>Community Detection|SVD,<br>Node2vec,<br>Community Detection Algs|-|ASR FPR |Reverse Engineered DGA Domains,<br>NXDOMAIN|-|

<a class="toc" id ="Type"></a>
# Attack Type
|Method|Targeted|Non-targeted|Black-box|White-box|Poisoning|Evasion|Topology|Feature|
|:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|[1] [Adversarial Attacks On Graph Neural Networks Via Meta Learning](#1)|❌|⭕|⭕|⭕|⭕|❌|⭕|❌|
|[2] [Adversarial Attacks On Node Embeddings Via Graph Poisoning](#2)|⭕|⭕|⭕|⭕|⭕|❌|⭕|❌|
|[3] [Attacking Graph Convolutional Networks Via Rewiring](#3)|❌|⭕|⭕|⭕|❌|⭕|⭕|❌|
|[4] [Topology Attack And Defense For Graph Neural Networks: An Optimization rspective](#4)|❌|⭕|⭕|⭕|⭕|❌|⭕|❌|
|[5] [Unsupervised Euclidean Distance Attack On Network Embedding](#5)|❌|⭕|⭕|⭕|⭕|❌|⭕|❌
|[6] [Generalizable Adversarial Attacks Using Generative Models](#6)|⭕|❌|❌|⭕|❌|⭕|❌|⭕
|[7] [Data Poisoning Attack Against Knowledge Graph Embedding](#7)|⭕|❌|⭕|❌|⭕|❌|⭕|❌|
|[8] [Attacking Graph-Based Classification Via Manipulating The Graph Structure](#8)|⭕|❌|⭕|⭕|⭕|❌|⭕|❌|
|[9] [Ga Based Q-Attack On Community Detection](#9)|
|[10] [$\alpha$cyber: Enhancing robustness of android malware detection system against adversarial attacks on heterogeneous graph based model](#10)|
|[11] [Data Poisoning Attacks on Neighborhood-based Recommender Systems](#11)|
|[12] [Adversarial Recommendation: Attack of the Learned Fake Users](#12)|
|[13] [Attack Graph Convolutional Networks By Adding Fake Nodes](#13)|⭕|⭕|❌|⭕|⭕|❌|⭕|⭕|
|[14] [Attack Tolerance Of Link Prediction Algorithms: How To Hide Your Relations In Social Network](#14)|⭕|❌|⭕|❌|❌|⭕|⭕|❌|
|[15] [Link Prediction Adversarial Attack](#15)|⭕|❌|⭕|⭕|❌|⭕|⭕|❌|
|[16] [Adversarial Attack On Graph Structured Data](#16)|⭕|❌|⭕|⭕|❌|⭕|⭕|❌|
|[17] [Adversarial Attacks On Neural Networks For Graph Data](#17)|⭕|❌|⭕|⭕|⭕|❌|⭕|⭕|
|[18] [Fast Gradient Attack On Network Embedding](#18)|⭕|❌|⭕|⭕|⭕|❌|⭕|❌|
|[19] [Data Poisoning Attack Against Unsupervised Node Embedding Methods](#19)|⭕|⭕|⭕|⭕|⭕|❌|⭕|❌|
|[20] [Attacking Similarity-Based Link Prediction In Social Networks](#20)|
|[21] [Practical Attacks Against Graph-Based Clustering](#21)|


![Metric](imgs/Attack.png)

# Defense

| Venue| Title | Model | Algorithm | Defense Type | Target Task | Target Model | Baseline | Metric | Dataset | Code | 
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| KDD 2019 | <a class="toc" id ="101"></a> [[1]](https://arxiv.org/abs/1906.12269) [🔙](#toc2) | GNN (trained with RH-U) | Robustness Certification, Objective Based | Hybrid | Node Classification | GNN, GCN | GNN (trained with CE, RCE, RH) | Accuracy, Averaged worst-case Margin | Citeseer, Cora-ML, Pubmed | [Link](https://www.kdd.in.tum.de/research/robust-gcn/) |
| IJCAI 2019 | <a class="toc" id ="102"></a> [[2]](https://arxiv.org/abs/1906.04214) [🔙](#toc2) | - | Adversarial Learning | Adversarial Learning | Node Classification | GCN | GCN | Misclassification Rate <br>Accuracy | Citeseer, Cora | [Link](https://github.com/KaidiXu/GCN_ADV_Train) |
| Arxiv 2019 | <a class="toc" id ="103"></a> [[3]](https://arxiv.org/abs/1905.10029) [🔙](#toc2) | r-GCN, VPN | Graph Powering | Objective Based | Node Classification | GCN | ManiReg, SemiEmb, LP, DeepWalk, ICA, Planetoid, Vanilla GCN | Accuracy, Robustness Merit,<br>Attack Deterioration | Citeseer, Cora, Pubmed | - |
| IJCAI 2019 | <a class="toc" id ="104"></a> [[4]](https://arxiv.org/abs/1903.01610) [🔙](#toc2) | - | Drop Edges | Preprocessing | Node Classification | GCN | GCN | Classfication Margin, Accuracy | Cora-ML, Citeseer, PolBlogs | [Link](https://github.com/stellargraph/stellargraph/) |
| Arxiv 2019 | <a class="toc" id ="105"></a> [[5]](https://arxiv.org/abs/1905.03679) [🔙](#toc2) | DefNet | GAN,<br>GER,<br>ACL | Hybrid | Node Classification | GCN, GraphSAGE | GCN, GraphSage | Classfication Margin | Cora, Citeseer, PolBlogs | - |
| Arxiv 2019 | <a class="toc" id ="106"></a> [[6]](https://arxiv.org/abs/1905.00563) [🔙](#toc2) | CRIAGE | Adversarial Modification | Robustness Evaluation | Link Prediction | Knowledge Graph Embedding | - | Hits@K, MRR | Nations, Kinship, WN18, YAGO3-10 | - |
| KDD 2019 | <a class="toc" id ="107"></a> [[7]](http://pengcui.thumedialab.com/papers/RGCN.pdf) [🔙](#toc2) | RGCN | Gaussian-based Graph Convolution | Structure Based | Node Classification | GCN | GCN, GAT | Accuracy | Cora, Citeseer, Pubmed | [Link](https://github.com/thumanlab/nrlweb/blob/master/static/assets/download/RGCN.zip) |
| Arxiv 2019 | <a class="toc" id ="108"></a> [[8]](https://arxiv.org/abs/1903.05994) [🔙](#toc2) | Global-AT, Target-AT, SD, SCEL | Adversarial Training, Smooth Defense | Hybrid | Node Classification | GNN | AT | ADR, ACD | PolBlogs, Cora, Citeseer | - |
| Arxiv 2019 | <a class="toc" id ="109"></a> [[9]](https://arxiv.org/abs/1902.11045) [🔙](#toc2) | SVAT, DVAT | Virtual Adversarial Training | Adversarial Training | Node Classification | GCN | GCN | Accuracy | Cora, Citeseer, Pubmed | - |
| RLGM@ICLR 2019 | <a class="toc" id ="110"></a> [[10]](https://rlgm.github.io/papers/57.pdf) [🔙](#toc2) | - | KL Divergence | Detection Based | Node Classification | GCN, GAT | - | Classfication Margin, Accuracy,<br>ROC, AUC | Cora, Citeseer, PolBlogs | - |
| Arxiv 2019 | <a class="toc" id ="111"></a> [[11]](https://arxiv.org/abs/1902.08226) [🔙](#toc2) | GCN-GATV | Graph Adversarial Training, Virtual Adversarial Training | Adversarial Training | Node Classification | GCN | LP, DeepWalk, SemiEmb, Planetoid, GCN, GraphSGAN | Accuracy | Citeseer, Cora, NELL | - |
| OpenReview 2018 | <a class="toc" id ="112"></a> [[12]](https://openreview.net/forum?id=HJxdAoCcYX) [🔙](#toc2) | SL, OD, GGD, LP+GGD, ENS | Link Prediction, Subsampling, Neighbour Analysis| Hybrid | Node Classification | GNN, GCN | LP | AUC | Cora, Citeseer | - |
| ICML 2019 | <a class="toc" id ="113"></a> [[13]](https://arxiv.org/abs/1902.09192) [🔙](#toc2) | S-BVAT, O-BVAT | Batch Virtual Adversarial Training | Adversarial Training | Node Classification | GCN | ManiReg, SemiEmb, LP, DeepWalk, Planetoid, Monet, GAT, GPNN, GCN, VAT | Accuracy | Cora, Citeseer, Pubmed, Nell | [Link](https://github.com/thudzj/BVAT) |
| Arxiv 2019 | <a class="toc" id ="114"></a> [[14]](https://arxiv.org/abs/1908.07558) [🔙](#toc2) | PA-GNN | Penalized Aggregation, Meta Learning | Structure Based | Node Classification | GNN | GCN, GAT, PreProcess, RGCN, VPN | Accuracy | Pubmed, Reddit, Yelp-Small, Yelp-Large | - |
| Arxiv 2019 | <a class="toc" id ="115"></a> [[15]](https://arxiv.org/abs/1911.04429) [🔙](#toc2) | GraphDefense | Adversarial Training | Adversarial Training | Node Classification | GCN | Drop Edges, Discrete Adversarial Training | Accuracy | Cora, Citeseer, Reddit | - |
| SIGIR 2018 | <a class="toc" id ="116"></a> [[16]](https://dl.acm.org/citation.cfm?id=3209981) [🔙](#toc2) | APR, AMF | Adversarial Training based on MF-BPR | Adversarial Training | Recommendation | MF-BPR | ItemPop, MF-BPR, CDAE, NeuMF, IRGAN | HR, NDCG | Yelp, Pinterest, Gowalla | [Link](https://github.com/hexiangnan/adversarial_personalized_ranking) |
| CIKM 2019 | <a class="toc" id ="117"></a> [[17]](https://dl.acm.org/citation.cfm?id=3209981) [🔙](#toc2) | Rad-HGC | HG-Defense | Detection Based | Malware Detection | Malware Detection System | FakeBank, CryptoMiner, AppCracked, MalFlayer, GameTrojan, BlackBaby, SDKSmartPush, ... | Detection Rate | Tencent Security Lab Dataset | - |

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


# Metric
![Metric](imgs/Metric.png)

