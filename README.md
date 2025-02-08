# AI_Project
I worked on an AI Project<br>

<p>
To get started I used Google Colab to import dataset from Kaggle 
To add Kaggle Dataset in Google Colab run the following commands

!pip install kaggle
from google.colab import files
files.upload()
import os
os.makedirs('/root/.kaggle', exist_ok=True)
os.rename('kaggle.json', '/root/.kaggle/kaggle.json')
!chmod 600 /root/.kaggle/kaggle.json
!kaggle datasets download -d alekseikrasnov/images-dataset-for-chemical-images-classifier
!unzip  /content/images-dataset-for-chemical-images-classifier.zip
</p>