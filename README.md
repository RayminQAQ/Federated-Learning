# PyTorch-FedLearning
A Pytorch and Tenserflow federated learning framework for CNNs. Trains on decentralized image data, emphasizing privacy and security. Features include client dataset distribution, local training, gradient aggregation, and epoch-wise accuracy visualization.

There are two files avaliable for user to clone and our project allow for customized dataset.
- Pytorch-federated.py
- tensorflow-federated.py

We ended up 97% accuracy in our customized dataset generated from our other repository.

[Snippet for out accuracy]
  - epsilon = 100 -> Accuracy: 0.11784865707159042
  - epsilon = 50 -> Accuracy: 0.16763243079185486
  - epsilon = 10 -> Accuracy: 0.09182702749967575
  - epsilon = 1e-06 -> Accuracy: 0.9792972803115845
  - epsilon = 1e-08 -> Accuracy: 0.9804324507713318
  - epsilon = 0 -> Accuracy: 0.9815567135810852

[Notice]: main.py 即將被刪出，因為模型準確率不好。
