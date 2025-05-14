# MNIST Handwritten Digit Classifier for Apple M1 (PyTorch)

This project implements a simple Convolutional Neural Network (CNN) using PyTorch to classify handwritten digits from the MNIST dataset. It is optimized to run efficiently on Apple Silicon (M1/M2/M3) chips using the MPS backend for GPU acceleration, while also providing fallback support for CUDA (NVIDIA GPUs) and CPU.

## Requirements

- Python 3.x
- PyTorch (ensure your installation supports MPS for Apple Silicon or CUDA for NVIDIA GPUs)
- Torchvision
- Matplotlib
- NumPy

## Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```
    *Note: Remember to replace `YOUR_USERNAME/YOUR_REPOSITORY_NAME` with the actual GitHub path if you are hosting this code there.*

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    *On Windows, use `.\.venv\Scripts\activate`*

3.  **Install dependencies:**
    You can install the required libraries directly:
    ```bash
    pip install torch torchvision torchaudio matplotlib numpy
    ```
    *Note: Ensure you install the correct PyTorch version for your specific hardware (MPS, CUDA, or CPU). Refer to the official PyTorch installation guide for details.*

## Running the Script

Once the setup is complete and your virtual environment is activated:

```bash
python mnist-cnn.py