[toc]
# Attack
## 1. Link Prediction Adversarial Attack
**Author**: Jinyin Chen, Ziqiang Shi, Yangyang Wu, Xuanheng Xu and Haibin Zheng
**Venue**: Arxiv, 2018, https://arxiv.org/abs/1810.01110
**Algorithm**: 
Iterative gradient attack (IGA) based on Graph auto-encoder (GAE)


**Attack Settings**:
+ **Unlimited attack**: Without any limitation on link modification, all the links decided by gradients are valid and the only limitation is the total number of modified links
+ **Single node attack**: A target link $E_t$ in the network is a connection of two nodes $(u; v)$, single node attack is defined to modify and only modify links connect to either node $u$ or $v$

**Target Task**: Link prediction

**Target Model**: GAE, DeepWalk, node2vec, CN, RA, Katz and Local Random Walk

**Baseline**:  
+ Random Attack (RAN)
+ Disconnect Internally, Connect Externally (DICE)
+ Gradient Attack (GA)

**Evaluation metric**
+ **Attack Success Rate (ASR)**: The ratio of the target links which will be successfully attack within $k_t$ modified links for each target versus all target links
+ **Average Modified Link number (AML)**: The average perturbation size leading to successful attack. For those failed attack, regard it as the total perturbation size $k_t$

**Code**: GAE: https://github.com/tkipf/gae

**Abstract**: This paper proposed a attack algorithm named *Iterative gradient attack* (IGA), which based on *graph auto-encoder* (GAE). It's the first time link prediction adversarial attack problem is defined and attack method is brought up.
To be more realistic, The authors add the following constraint to the attack algorithm: *Unlimited attack* and *Single node attack*. One thing they have in common is the limitation is the total number of modified links, however, in unlimited attack, all the links decided by gradients are valid(means could be modified), while can only modify links adjacent to target link in single node attack.
The deatils of the algorighm are described as follows:
+ Choose one link as the target to make predictors cannot precisely work
+ Calculate gradient for this target to generate a corresponding adversarial network iteratively
+ The adversarial network and the original clean network are adopted as the input for various predictors

These predictors will wrongly predict the target in adversarial network while correctly predict it in the original clean network to implement the adversarial attack.

---


## 2. Adversarial Attack on Graph Structured Data
**Author**: Hanjun Dai, Hui Li, Tian Tian, Xin Huang, LinWang, Jun Zhu, Le Song
**Venue**: ICML, 2018, https://arxiv.org/abs/1806.02371
**Algorithm**: 
+ RL-S2V (Reinforcement Learning structure2vec)
+ GradArgmax (Gradient Based White Box Attack)
+ GeneticAlg (Genetic Algorithm)

**Attack Settings**:
+ **White box attack (WBA)**: The attacker is allowed to access any information of the target classifier, including the prediction, gradient information, etc..
+ **Practical black box attack (PBA)**: Only the prediction of the target classifier is available.
    + **PBA-C**: The prediction confidence is accessible
    + **PBA-D**: Only the discrete prediction label is allowed
+ **Restrict black box attack (RBA)**: Further than PBA, only do black-box queries on some of the samples, and the attacker is asked to create adversarial modifications to other samples

**Target Task**: 

**Graph classification**
+ Graph-level attack
+ Node-level attack

**Target Model**: Graph Neural Network

**Baseline**:  
+ **Random Sampling**


**Evaluation metric**
+ **Accuracy**

**Code**: https://github.com/Hanjun-Dai/graph_adversarial_attack
**Abstract**: This paper first propose a reinforcement learning based attack method(RL-S2V) that learns the generalizable attack policy, while only requiring prediction labels Vebue the target classifier. And further propose attack methods based on genetic algorithms and gradient descent in the scenario where additional prediction confidence or gradients are available. These algorithm are used to attack a family of Graph Neural Network models, and the Target Task divided into graph-level attack and node-level attack, finally show great results in real-world data. Besides, this paper also proposed a cheap method for defencing: simply
doing edge drop during training for defense.

---

## 3. Adversarial Attacks on Neural Networks for Graph Data
**Author**: Daniel Zügner, Amir Akbarnejad, Stephan Günnemann
**Venue**: SIGKDD, 2018, https://arxiv.org/abs/1805.07984
**Algorithm**: 
+ Nettack based on GCN


**Attack Settings**:
+ Type 1
    + **Evasion**: The model parameters are kept fix based on the clean graph, test-time attacks
    + **Poisoning attacks**: The model is retrained after the attack, training-time attacks

+ Type 2
    + **Influencer attack**: if the target $v_0$ not belongs to the set of attacker nodes.
    + **Direct attack**: if the target $v_0$ belongs to the set of attacker nodes, i.e., $v_0$ gets manipulated directly.

+ Type 3
    + **Structure attacks**: Change the adjective matrix
    + **Feature attacks**:  Change the feature matrix


**Target Task**: **Node classification**

**Target Model**: 
+ Semi-supervised methods: GCN(Graph Convolutional Network), CLN(Column Network)
+ Unsupervised model DeepWalk

**Baseline**:  
+ **Random Sampling**
+ **Fast Gradient Sign Method (FGSM)**


**Evaluation metric**
+ **Classification margin**:
$$ X=Z_{v_{0}, c_{o l d}}^{*}-\max _{c \neq c_{o l d}} Z_{v_{0}}^{*} $$
Using the ground truth label $c_{old}$ of the target, The smaller $X$, the better.


**Code**: https://github.com/danielzuegner/nettack
**Abstract**: This paper presented the first work on adversarial attacks to (attributed) graphs, specifically focusing on the task of node classification via graph convolutional networks. The attacks target the nodes’ features and the graph structure, and there are two types of attacks: direct and influencer attacks.The classification performance is consistently reduced, even when only partial knowledge of the graph is available or the attack is restricted to a few influencers. The detail of the algorithm is:
+ Train the surrogate model on the labeled data and among all nodes Vebue the test set that have been correctly classified.
+ Given perturbations, selected candidate links and features based on their important
data characteristics such as degree distribution and cooccurence of features. 
+ Defined two scoring functions to evaluate the change in the confidence value of the target node after modifying a link and feature in the candidate sets, respectively. 
+ Used the link or feature of the highest score to update the adversarial network
+ For the final changed graph, train the model on it and calculate the *Classification margin*.

 However, this approach is limited to node classification task, with little discussion on the transferability of the attack.

---
## 4. Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective
**Author**: Kaidi Xu1, Hongge Chen2， Sijia Liu3, Pin-Yu Chen3, Tsui-Wei Weng2, Mingyi Hong4 and Xue Lin1
**Venue**: IJCAI, 2019, https://arxiv.org/abs/1906.04214
**Algorithm**: 
+ Attack
    + Projected gradient descent (PGD) topology attack 
        + Negative cross-entropy loss via PGD attack (CE-PGD)
        + Carlili-Wagner loss via PGD attack (CW-PGD)
    + Min-max topology attack
        + Negative cross-entropy loss via min-max attack (CE-minmax)
        + Carlili-Wagner loss via min-max attack (CW-min-max)
+ Defence
    + Optimization-based adversarial training technique


**Attack Settings**:
+ Topology Attack in Terms of Edge Perturbation (Add/Drop edges)
    + Attacking a pre-defined GNN 
    + Attacking a re-trainable GNN


**Target Task**: **Node classification**

**Target Model**: 
+ Graph convolutional methods: GCN


**Baseline**:  
+ **DICE (Delete Edges Internally, Connect Externally)**
+ **Meta-Self attack**
+ **Greedy attack(a variant of Meta-Self attack without weight re-training for GCN)**


**Evaluation metric**
+ **Misclassification rate**:
$mean \pm standard\ deviation$ of misclassification rate (namely, 1 - prediction accuracy)on testing nodes.


**Code**: None
**Abstract**: This paper first present a novel gradient-based topology attack framework that overcomes the difficulty of attacking discrete graph structure data a first-order optimization perspective, the attack framework which under two attacking scenarios:
+ Attacking a pre-defined GNN
+ Attacking a re-trainable GNN

With two topology attack methods: projected gradient descent (PGD) topology attack and min-max topology attack.
Besides, With the aid of the first-order attack generation methods, this paper then propose an adversarial training method for GNNs to improve their robustness.

## 5. Fast Gradient Attack on Network Embedding
**Author**: Jinyin Chen, Yangyang Wu, Xuanheng Xu, Yixian Chen, Haibin Zheng, and Qi Xuan IEEE Member
**Venue**: Arxiv, 2018, https://arxiv.org/abs/1809.02797
**Algorithm**: 
+ (FGA) Fast Gradient Attack based on GCN

**Attack Settings**:
+ Type 1
    + **Direct attack**: Attack the links around the target nodes
    + **Indirect attack**: Attack the links that are not immediately connected to the target nodes
    + **Unlimited attack**: Do not limit the attack to the direct or indirect links, i.e., can remove or add a link between any pair of nodes
+ Type 2
    + **White-Box Adversarial Attack**
    + **Black-Box Adversarial Attack**
+ Type 3
    + **Hub-node attack**: Choose some nodes of largest degree in each network as target nodes
    + **Bridge-node attack**: Choose nodes of highest betweenness centrality (connecting different communities) as target nodes

**Target Task**: 
+ **Node classification**
+ **Community detection**

**Target Model**: 
GCN, GraRep, DeepWalk, Node2vec, LINE and GraphGAN


**Baseline**:  
+ **Random Attack (RA)**: RA randomly disconnects $b (b < K)$ links in the original network, while randomly connects $K - b$ pairs of nodes that are originally not connected
+ **Disconnect Internally, Connect Externally (DICE)**:DICE first randomly disconnect b links of target node, then randomly connect the target node to $K - b$ nodes of different classes.
+ **NETTACK**

**Evaluation metric**
+ **Attack Success Rates (ASR)**
+ **Average number of Modified Links (AML)**

**Code**: None
**Abstract**: 
This paper propose a framework to generate adversarial networks FGA using GCN, the detail of the algorithm are described as follows:
+ Design an adversarial network generator, utilizing the iterative gradient information of pairwise nodes based on the trained GCN model to generate adversarial network so as to realize the FGA
+ Use FGA to attack not only GCN model but also several other network embedding methods, such as GraRep, DeepWalk, node2vec, LINE and Graph-GAN, and propose various adversarial attack strategies
to attack multiple network analysis methods
+ This iterative process is terminated when the number of modified links reaches certain predefined value
The results suggest that, in any case, FGA outperform the other baseline attack methods, achieving the state-of-the-art results in Node classification and Community detection

This paper is very similar to Link Prediction Adversarial Attack

---

## 6. Data Poisoning Attack against Unsupervised Node Embedding Methods
**Author**: Mingjie Sun, Jian Tang, Huichen Li, Bo Li, Chaowei Xiao, Yao Chen, Dawn Song
**Venue**: Arxiv, 2018, https://arxiv.org/abs/1810.12881
**Algorithm**: 
+ Projected Gradient Descent (PGD)

**Attack Settings**:
+ **Integrity attack**: The attacker’s goal is either to increase or decrease the probability (similarity score) of a target node pair
    + **Direct Attack**: The attacker can only manipulate edges adjacent to the target node pair
    + **Indirect Attack**: The attacker can only manipulate edges without connecting to the target node pair.
+ **Availability attack**: The attacker’s goal is to reduce the prediction performance over a test set


**Target Task**: 
+ **Link prediction**

**Target Model**: 
+ **white box**: DeepWalk, LINE
+ **black box**: Node2vec, SC(Spectral Clustering), GAE


**Baseline**:  
+ **Random Attack (RA)**: Randomly add or remove edges
+ **Personalized PageRank**:  Given a target edge (A;B), use personalized PageRank to calculate the importance of the nodes. Given a list of nodes ranked by their importance, e.g., (x1; x2; x3; x4; ...), we select the edges which connect the top ranked nodes to A or B, i.e., (A; x1); (B; x1); (A; x2); (B; x2); ...
+ **Degree sum**: Rank the node pair by the sum of degree of its two nodes, Then add or delete the node pair with the largest degree sum.
+ **Shortest path**: Rank the edge by the number of times that it is on the shortest paths between
two nodes in graph. Then delete the important edges measured by the number of shortest paths
in graph that go through this edge.

**Evaluation metric**
+ **Similarity score increase/decrease**
+ **Average precision score (AP score)**

**Code**: None
**Abstract**: This paper propose a unified optimization framework to optimally attack the node embedding methods by adding or removing a few edges, for both types of attacks, including integrity attack and availability attack. The attack method Projected Gradient Descent (PGD), which is based on two popular node embedding methods: DeepWalk and LINE. The detail of the algorithm are described as follows:
+ Projected Gradient Descent (PGD) step: Calculate gradient of the weighted adjacency matrix based on DeepWalk and LINE, respectively
+ Projection step: projection of weighted adjacency matrix onto $\{0, 1\}^{|V| \times |V|}$
if the cell in the matrix is cloest to 1 and the original value is 0, add edges, similarly, delete edges that cloest to  0 and the original value is 1.
However, the projection step has two choice:
+ project once after the projected gradient descent step
+ incorporate the projection step into the gradient descent computation where we project every $k$ iterations

Besides, this paper conduct a case study on a coauthor network to better understand the attack method, and have the following observations:
+ The edges chosen by attack methods mainly lies on shortest path between target pair nodes.
+ The edges chosen by attack methods are mostly cross-field edges

## 7. Adversarial attacks on graph neural networks via meta learning
**Author**: Daniel Zügner, Stephan Günnemann
**Venue**: ICLR, 2019, https://arxiv.org/abs/1902.08412
**Algorithm**: 
+ Meta-Self (With self-training, $\lambda = 0$)
+ Meta-Train (Variants without self-training, $\lambda = 1$)
+ Meta-Both (Using both, $\lambda = 0.5$)

Based on GCN

**Attack Settings**:
+ Global attack: To reduce the overall classification performance of a model at training time.


**Target Task**: 
+ **Node classification**

**Target Model**: 
GCN, CLN, DeepWalk


**Baseline**:  
+ **Disconnect Internally, Connect Externally (DICE)**: For each perturbation, randomly choose whether to insert or remove an edge. Edges are only removed between nodes from the same class, and only inserted between nodes from different classes. This baseline has all true class labels (train and test) available
+ **Nettack**
+ **First-order**

**Evaluation metric**
+ **Misclassification rate**: 1 - accuracy on unlabeled nodes obtained on the test data

**Code**: https://www.kdd.in.tum.de/research/gnn-meta-attack/

**Abstract**: This paper propose an algorithm for training-time adversarial attacks on (attributed)graphs, focusing on the task of node classification using meragradients. For efficency, this paper further propose approximations of the metagradients that are less expensive to compute. The algorithm is similar to NETTACK, calculate the score $S$ of each edge, and select the edge that with highest score. However, the difference is that, using metagradients to calculate $S$.


## 8. Adversarial Attacks on Node Embeddings via Graph Poisoning
**Author**: Aleksandar Bojchevski, Stephan Günnemann
**Venue**: ICML, 2019, https://arxiv.org/abs/1809.01093
**Algorithm**: 
+ $\mathcal{A}_{DW2}$: Gradient based attack
+ $\mathcal{A}_{DW3}$: Closed-form attack
+ $\mathcal{A}_{link}$: Targeted link prediction attack
+ $\mathcal{A}_{class}$: Targeted node classification attack
Based on random walks (RWs)

**Attack Settings**:
+ **General Attack**: nontargeted attack
    + **Unrestricted attacks**
    + **Restricted attacks**: Given percentage $p_r$ of nodes as restricted and do not use them to attack
+ **Targeted Attack**


**Target Task**: 
+ **Node classification**
+ **Link prediction**

**Target Model**: 
DeepWalk


**Baseline**:  
+ **$\mathcal{B}_{rnd}$**: Randomly flips edges
+ **$\mathcal{B}_{eig}$**: Removes/add edges based on their eigencentrality in the line graph L(A)
+ **$\mathcal{B}_{deg}$**: Removes/add edges based on their degree centrality in L(A)


**Evaluation metric**
+ **Change in F1 score** for node classification
+ **Classification margin** for node classification
+ **Change in AP score (average precision score)** for link prediction
+ **Pearson’s R score** between the actual loss to compare algorithm

**Code**: https://www.kdd.in.tum.de/research/node_embedding_attack/

**Abstract**: This paper provide the first adversarial vulnerability analysis on the widely used family of node embedding methods based on random walks. By exploiting results from eigenvalue perturbation theory, they can efficiently calculate the loss when graph changed, and avoids
recomputing the eigenvalue decomposition. For node classification and link prediction, they assume an attacker with full knowledge about the data and the model.
The details of the algorithm are describe as follows:
+ For general attack:
    + Form a candidate set by randomly sampling $C$ candidate pairs
    + For every candidate we compute its impact on the loss and greedily choose the top $f$ flips
+ For targeted attack:
    + Form a candidate set directly incident to the target node
    + Pre-train a classifier on the clean embedding, and predict the class probabilities, select the top $f$ flips with smallest classification margin. (Targeting node classification)
    + First train an initial clean embedding on the graph, calculate the average precision score, pick the top $f$ flips with lowest AP scores (Targeting link prediction)