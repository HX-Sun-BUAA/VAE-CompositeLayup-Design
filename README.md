This code for the paper: Efficient property-oriented design of composite layups via controllable latent features using generative VAE
![Graph Abstract](https://github.com/user-attachments/assets/9b51d968-0ed4-470f-bc41-3536284de673)

# Project Overview
This project contains code used in my thesis research on designing laminate composites using a Variational Autoencoder (VAE) model. The code is implemented in PyTorch and includes training and testing routines for generating layups with desired laminate parameters, performing design evaluations, and ensuring compliance with design constraints.

# Contents
main_code.ipynb: The main Jupyter notebook file containing all the code for training the VAE model, generating designs, and evaluating laminate properties.
Pre-trained Model: The repository includes code for saving and loading a pre-trained VAE model (0304vae_model.pth).
Ply Angle Data: Datasets for laminate ply angles and encoded laminate parameters are also referenced (0304ply_angle.pt, 0304normalized_LP_tensor.pt, etc.).

# Requirements
To run this code, you will need the following:

Python 3.x
PyTorch
NumPy
Matplotlib
Joblib
Pandas

# Usage
1.Training the VAE Model: The notebook contains code to train the VAE model for designing laminate structures. You can modify the training parameters and dataset paths based on your requirements.

2.Design Generation: Once trained, the VAE model can be used to generate laminate designs based on the input specifications. This part of the notebook demonstrates how to normalize target laminate parameters, generate random components, and produce designs using the VAE.

3.Evaluating Designs: The notebook also provides functionality to evaluate the generated designs by calculating laminate parameters, ensuring compliance with design constraints, and visualizing results.

# Results and Visualizations
Ply angles and laminate parameters are visualized using Matplotlib.
Validation and training losses, as well as design accuracy, are plotted to monitor the performance of the VAE model.

# How to Run
Download the repository and run the notebook in Jupyter or a similar environment.
Ensure that all dataset files are available in the same directory as the notebook.
Follow the steps in the notebook to either train the model or load a pre-trained model to generate and evaluate designs.

# To load the pre-trained VAE model:
model = VAE()  # Initialize your model class

model.load_state_dict(torch.load('dataset/0304vae_model.pth'))  # Load weights

# Contact
For any questions or issues, feel free to open an issue on this repository or contact me directly via GitHub.
