🚦 Traffic Sign Classification using Deep Learning.


📌 Project Overview
Traffic signs play a crucial role in ensuring road safety and guiding drivers. This project focuses on building a Traffic Sign Classification System using Convolutional Neural Networks (CNNs).
The model classifies images into 43 different categories of traffic signs and achieves state-of-the-art accuracy (~99%) on the test dataset.

📂 Dataset:-

Source: German Traffic Sign Recognition Benchmark (GTSRB)
Classes: 43 different traffic signs
Image Size: Resized to 32×32×3
Total Images: ~50,000


Example signs include:
Speed limit signs
Stop signs
Pedestrian crossings
No entry signs
Yield signs, etc.

⚙️ Model Architecture
We used a Convolutional Neural Network (CNN) built with TensorFlow/Keras.


Architecture Highlights:

Input Layer: 32×32×3 RGB images
Conv2D + MaxPooling layers for feature extraction
Dropout layers to reduce overfitting
Fully Connected Dense layers
Output Layer: 43 neurons (Softmax activation)

Optimizer: Adam
Loss Function: Categorical Cross-Entropy
Batch Size: 32
Epochs: 15

📊 Training Results
Accuracy & Loss Curves
Training Accuracy	Validation Accuracy	Training Loss	Validation Loss
📈 Both training and validation accuracy converge to ~99%, showing strong generalization.

🎯 Performance Metrics
Test Accuracy: 99%
Macro Avg Precision / Recall / F1-score: 0.99
Weighted Avg Precision / Recall / F1-score: 0.99



Classification Report (Sample)
precision    recall  f1-score   support

0     1.00      0.96      0.98        27
1     1.00      0.99      0.99       291
2     0.99      0.99      0.99       290
3     0.96      1.00      0.98       203
...
42    0.98      1.00      0.99        42

Confusion Matrix (Highlights)

✔️ Most predictions are correctly classified with minimal misclassifications.

🚀 How to Run
Clone this repository:

git clone https://github.com/your-username/traffic-sign-classification.git
cd traffic-sign-classification


Install dependencies:
pip install -r requirements.txt


Run training:

python train.py


Test model:
python evaluate.py


🔮 Future Work

Deploy the model in a real-time traffic monitoring system.
Integrate with self-driving car simulation environments.
Optimize with transfer learning (ResNet, MobileNet) for faster inference.
Build a mobile/web app for live traffic sign detection.

📌 Conclusion

This project demonstrates that a CNN can accurately classify traffic signs with 99% accuracy, making it suitable for applications in autonomous driving and road safety systems.
