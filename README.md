# Local rotation invariance in 3D CNNs
Implementation of the 3D Locally Rotation Invariant CNNs developed in "Local rotation invariance in 3D CNNs", Vincent Andrearczyk, Julien Fageot, Valentin Oreiller, Xavier Montet, Adrien Depeursinge. Paper submitted to MedIA.

# How to use it
To obtain LRI, the first convolution of a 3D CNN can be replaced by one of the LRI variants: 
- s_conv3d, defined in ./sh_networks_utils.py is obtained by steering responses to a set of orientation and locally taking the maximum response.
- sse-conv3d, also defined in ./sh_networks_utils.py is obtained by calculating invariants in the form of Solid Spherical Energy (SSE) from the response to convolutions with spherical harmonics.

Example usage is provided on a basic synthetic texture dataset in ./synthetic_experiments/


# Installation and Requirements
The environment needed to run the experiments must include Tensorflow 1.8.0

The list of python dependecies is provided in requirements.txt and can be installed by running:
pip install -r requirements.txt
