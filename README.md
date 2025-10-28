README: Deep Learning Image Classification Project
This repository contains the code, report, and presentation for a deep learning project focused on classifying the Animals-10 dataset. The project explores different neural network architectures using a refractory method to achieve high accuracy in image classification, which involved fine-tuning pre-trained models.

Repository Files
Jupyter Notebooks
This project uses three main Jupyter notebooks to implement different models: a custom Convolutional Neural Network (CNN) and two models for transfer learning, MobileNetV2 and ResNet50.

G3_CNN.ipynb: This notebook implements a custom CNN model from scratch. The architecture consists of multiple Conv2D layers, MaxPooling layers, and a Dropout layer to prevent overfitting. The model is trained on the Animals-10 dataset after preparing the data through resizing and normalization. It also includes code for visualizing the training and validation accuracy and loss plots, and it concludes with a Gradio interface for a simple image classifier.

G3_MobileNet.ipynb: This notebook leverages transfer learning with the MobileNetV2 model, which is pre-trained on the ImageNet dataset. The notebook details the process of fine-tuning the model for the Animals-10 dataset. This involved two main phases: first, freezing the base layers and training only the top layers, and second, unfreezing a few of the base layers for further training to improve performance. The code also includes model evaluation metrics such as accuracy, precision, recall, and F1-score.

G3_ResNet50.ipynb: This notebook also uses transfer learning but with the ResNet50 model. Similar to the MobileNetV2 notebook, it outlines the process of fine-tuning this powerful architecture on the Animals-10 dataset. The notebook shows how fine-tuning significantly improves both accuracy and loss on the training and validation sets, achieving a high accuracy of 98%. The confusion matrix and a detailed classification report are also generated to evaluate the model's performance on a per-class basis.

Additional Project Documents
G3_Report_DL_Image_Classification_with_CNN.pdf: This comprehensive report provides a detailed overview of the project, including the approach, methodology, and a thorough analysis of the results. It discusses the performance of each model and the impact of fine-tuning.

G3_Project_I_DL_presentation.pdf: This presentation provides a visual summary of the project's key findings. It covers the dataset selection, the architecture of each model, and the final results in a concise format.

# Author - Abhishek Thummanapelli