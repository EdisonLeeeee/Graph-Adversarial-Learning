[toc]
# Attack
|Year|Title|Model|Algorithm|Target Task|Target Model|Baseline|Evaluation Metric|Dataset|Venue|Code|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|2018|[Link Prediction Adversarial Attack](https://arxiv.org/abs/1810.01110)|IGA|Gradient based on GAE|Link prediction|GAE<br>DeepWalk<br>node2vec<br>CN<br>RA<br>Katz<br>LRW|RAN<br>DICE<br>GA|ASR(+)<br>AML(-)|NS<br>Yeast<br>FaceBook|Arxiv|None
|2018|[Adversarial Attack on Graph Structured Data](https://arxiv.org/abs/1806.02371)|RL-S2V|Reinforcement Learning|Graph classification|GCN|Random Sampling|Accuracy(-)|Citeseer<br>Cora<br>Pubmed<br>Finance|ICML|[Link](https://github.com/Hanjun-Dai/graph_adversarial_attack)
|2018|[Adversarial Attacks on Neural Networks for Graph Data](https://arxiv.org/abs/1805.07984)|Nettack|Based on GCN|Node classification|GCN<br>CLN<br>DeepWalk|Rnd<br>FGSM|Classification margin(-)<br>Accuracy(-)|Cora-ML<br>CiteSeer<br>PolBlogs|SIGKDD|[Link](https://github.com/danielzuegner/nettack)|
|2019|[Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective](https://arxiv.org/abs/1906.04214)|CE-PGD<br>CW-PGD|Gradient based|Node classification|GCN|DICE<br>Meta-Self attack<br>Greedy attack|Misclassification rate(+)|Cora<br>Citeseer|IJCAI|None|
|2018|[Fast Gradient Attack on Network Embedding](https://arxiv.org/abs/1809.02797)|FGA|Gradient based on GCN|Node classification<br>Community detection|GCN<br>GraRep<br>DeepWalk<br>Node2vec<br>LINE<br>GraphGAN|RA<br>DICE<br>NETTACK|ASR(+)<br>AML(-)|Cora<br>CiteSeer<br>PolBlogs|Arxiv|None|
|2018|[Data Poisoning Attack against Unsupervised Node Embedding Methods](https://arxiv.org/abs/1810.12881)|Opt-attack|Gradient based on DeepWalk,  LINE|Link prediction|DeepWalk<br>LINE<br>Node2vec<br>SC<br>GAE|RA<br>PageRank<br>Degree sum<br>Shortest path|Similarity score increase/decrease(+)<br>AP(-)|Facebook<br>Cora<BR>Citeseer|Arxiv|None|
|2019|[Adversarial attacks on graph neural networks via meta learning](https://arxiv.org/abs/1902.08412)|Meta-Self<br>Meta-Train|Meta Gradient based on GCN|Node classification|GCN<br>CLN<br>DeepWalk|DICE<br>NETTACK<br>First-order|Misclassification rate(+)|Cora<br>CiteSeer<br>PolBlogs|ICLR|[Link](https://www.kdd.in.tum.de/research/gnn-meta-attack/)|
|2019|[Adversarial Attacks on Node Embeddings via Graph Poisoning](https://arxiv.org/abs/1809.01093)|$\mathcal{A}_{DW2}$<br>$\mathcal{A}_{DW3}$|Gradient based on random walk|Node classification<br>Link prediction|DeepWalk|$\mathcal{B}_{rnd}$<br>$\mathcal{B}_{eig}$<br>$\mathcal{B}_{deg}$|Change in F1 score(+)<br>Classification margin(-)<br>|Cora<br>Citeseer<br>PolBlogs|ICML|[Link](https://www.kdd.in.tum.de/research/node_embedding_attack/)
|2019|[Attacking Graph Convolutional Networks via Rewiring](https://arxiv.org/abs/1906.03750)|ReWatt|Reinforcement learning based on GCN|Graph classification|GCN|RL-S2V<br>Random<br>Random-S|Success rate(+)|REDDIT-MULTI-12K<br>REDDIT-MULTI-5K<br>IMDB-MULTI|Arxiv|None|




