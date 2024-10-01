## Lung Cancer Detection Using CNN
This project aims to classify lung tissue as normal or cancerous using a Convolutional Neural Network (CNN). The model is trained on histopathological images from the [Kaggle Dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images). The dataset includes images of three classes:
- **Normal Tissue**
- **Lung Adenocarcinomas**
- **Lung Squamous Cell Carcinomas**

The model utilizes Python libraries like TensorFlow, Keras, OpenCV, and Scikit-learn to preprocess the images and build the classifier.

## Project Structure
- **Lung_Cancer_Detection.ipynb**: The Jupyter Notebook containing code for data processing, visualization, model building, training, and evaluation.
- **README.md**: Project documentation (you're reading it right now).

## Dataset
The dataset contains 5000 histopathological images classified into three classes. The dataset can be downloaded from Kaggle: [Lung and Colon Cancer Histopathological Images](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images).

## Installation
1. **Clone the Repository**:
   \`\`\`bash
   git clone https://github.com/SAI-ADITH/Lung-Cancer-Detection.git
   cd Lung-Cancer-Detection
   \`\`\`

2. **Install Python Dependencies**:
   Make sure you have Python and pip installed. Install the required packages by running:
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`
   If \`requirements.txt\` is not available, manually install the key packages:
   \`\`\`bash
   pip install numpy pandas matplotlib scikit-learn opencv-python tensorflow keras pydot graphviz
   \`\`\`

3. **Jupyter Notebook**: 
   To run the code, you need to have Jupyter Notebook installed. You can install it via:
   \`\`\`bash
   pip install notebook
   \`\`\`

## How to Run the Code
1. **Open Jupyter Notebook**:
   In your terminal, navigate to the project directory and run:
   \`\`\`bash
   jupyter notebook
   \`\`\`
   This will open a local server in your default web browser.

2. **Load the \`Lung_Cancer_Detection.ipynb\` Notebook**: 
   In the Jupyter Notebook interface, navigate to and open \`Lung_Cancer_Detection.ipynb\`.

3. **Execute the Cells**:
   - Follow the instructions provided in the notebook to run each code cell sequentially.
   - Make sure the dataset is placed correctly as per the directory structure outlined above.

## Model Architecture
The Convolutional Neural Network (CNN) model consists of the following layers:
1. **Convolutional Layers**: Three sets of Conv2D and MaxPooling2D layers.
2. **Flatten Layer**: Converts the 2D matrix output to a 1D vector.
3. **Dense Layers**: Two dense layers, one with \`BatchNormalization\` and \`Dropout\` for regularization.
4. **Output Layer**: A dense layer with \`softmax\` activation to classify the image into one of three classes.

The model is compiled using:
- **Optimizer**: \`Adam\`
- **Loss Function**: \`Categorical Crossentropy\`
- **Metrics**: \`Accuracy\`

## Results
The model achieved an F1 score of over 90% for each class, indicating a high level of accuracy in predicting the presence of cancerous cells in lung tissues.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
" > README.md
