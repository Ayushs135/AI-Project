# AI_Project
I worked on an AI Project<br>

<p>
This repository contains a Python script to classify images using a fine-tuned ResNet-18 model. The model is trained on a dataset with four classes and can be used to predict the class of a given image.<br>
The Dataset can be found here https://www.kaggle.com/datasets/alekseikrasnov/images-dataset-for-chemical-images-classifier<br>
To get started I used Google Colab to import dataset from Kaggle 
To add Kaggle Dataset in Google Colab run the following commands

!pip install kaggle<br><br>
from google.colab import files<br>
files.upload()<br><br>
import os<br>
os.makedirs('/root/.kaggle', exist_ok=True)<br>
os.rename('kaggle.json', '/root/.kaggle/kaggle.json')<br>
!chmod 600 /root/.kaggle/kaggle.json<br><br>
!kaggle datasets download -d alekseikrasnov/images-dataset-for-chemical-images-classifier<br><br>
!unzip  /content/images-dataset-for-chemical-images-classifier.zip<br><br>
</p>