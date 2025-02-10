# AI_Project
I worked on an AI Project<br>

<p>
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