# ModelTree
Collects implementations to identify model ancestry trees from model features. Currently only MoTHer, may be extended in the future.


# MoTHeR

This project implements the ideas presented in the paper "On the Origin of Llamas: Model Tree Heritage Recovery" [arxiv](https://arxiv.org/pdf/2405.18432) [github](https://github.com/eliahuhorwitz/MoTHer) in a simplified form. 
The code is structured to handle clustering of models, build minimal spanning trees, and compare the resulting adjacency matrix with a ground truth matrix.
Mostly, it ommits an explicit graph structure for simplicitly and instead relies on adjacency matrices.

## Structure
```
MoTHer/  
│  
├── models/  
│   └── model_operations.py        # Contains functions for loading models, manipulating state dictionaries, and adding noise.  
│
├── utils/
│   ├── helpers.py                 # General helper functions like kurtosis computation, distance calculation, and tree construction.
│   └── clustering.py              # Core clustering functions including hierarchical clustering and global tree construction.
│
├── example.py                     # Example script demonstrating the full workflow: loading models, adding noise, clustering, and evaluation.
├── README.md                      # This documentation file.
└── requirements.txt               # List of dependencies required to run the project.
```

## Dependencies

- `torch`
- `torchvision`
- `scipy`
- `numpy`
- `networkx`

All dependencies are listed in `requirements.txt`.

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd refactored_project
   ```
2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

### Section 5: Usage

## Usage
To use the code and test how well MoTHer identifies model relation, you'll have to build a model relation adjacency matrix. To get an idea of the interfaces, the `example.py` script demonstrates the entire process:

   - Loading a pre-trained Vision Transformer (ViT) model.
   - Creating noisy variations of the model's state dictionary.
   - Performing hierarchical clustering based on the layerwise distances.
   - Building minimal spanning trees for each cluster.
   - Constructing a global adjacency matrix and comparing it with a ground truth matrix.

   To run the script:

   ```bash
   python example.py

## Reference

Please refer to the paper ["On the Origin of Llamas: Model Tree Heritage Recovery"](https://arxiv.org/pdf/2405.18432) for the theoretical background and original ideas.
