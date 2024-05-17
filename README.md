# PyTorch-FedLearning
A Pytorch and Tenserflow federated learning framework for CNNs. Trains on decentralized image data, emphasizing privacy and security. Features include client dataset distribution, local training, gradient aggregation, and epoch-wise accuracy visualization.

There are two files avaliable for user to clone and our project allow for customized dataset.
- Pytorch-federated.py
- tensorflow-federated.py

We ended up 97% accuracy in our customized dataset generated from our other repository.

[Snippet for out accuracy]
  - epsilon = 100 -> Accuracy: 11.78 %
  - epsilon = 50 -> Accuracy: 16.76 %
  - epsilon = 10 -> Accuracy: 9.18 %
  - epsilon = 1e-06 -> Accuracy: 97.92 %
  - epsilon = 1e-08 -> Accuracy: 98.04 %
  - epsilon = 0 -> Accuracy: 98.15 %

[Notice]: main.py 即將被刪出，因為模型準確率不好。
