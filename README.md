# Image Classification Model Comparison

## Introduction

This repository features a comparison of different models trained for image classification on a bean
disease image dataset. The models include:

- Basic Machine Learning (ML) algorithms
- Convolutional Neural Network (CNN)
- Hybrid approach 1: CNN for feature extraction + ML algorithms for classification
- Hybrid approach 2: Stacked classification models with a final classification model for prediction

## Models and Methodology

1. Basic ML Algorithms

   - Logistic Regression
   - K-Nearest Neighbors
   - Support Vector Machine
   - Naive Bayes

2. CNN Model

   - Architecture: Transfer Learning using ResNet50 pretrained model

3. Hybrid Approach

   - Feature Extractors: CNNs (VGG16, VGG19, ResNet50) were used to extract features from images and ML
     algorithms were used for image classification using the extracted features.
   - Classifiers: Different ML algorithms including Logistic Regression, Naive Bayes, SVM, KNN...e.t.c.

## Evaluation Metrics

The models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score

## Results on Test Data

| Model                                             | Accuracy | Precision | Recall | F1-Score |
| ------------------------------------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression                               | 0.66     | 0.66      | 0.66   | 0.66     |
| K-Nearest Neighbors                               | 0.54     | 0.63      | 0.55   | 0.54     |
| Support Vector Machine                            | 0.68     | 0.68      | 0.68   | 0.68     |
| Naive Bayes                                       | 0.60     | 0.61      | 0.60   | 0.60     |
| CNN (ResNet50)                                    | 0.98     | 0.98      | 0.98   | 0.98     |
| Hybrid Model (Resnet50 & KNN)                     | 0.70     | 0.72      | 0.71   | 0.67     |
| Hybrid Model (VGG16 & Logistic Regression)        | 0.91     | 0.91      | 0.91   | 0.91     |
| Hybrid Model (VGG16 & SVM)                        | 0.84     | 0.86      | 0.84   | 0.84     |
| Hybrid Model (VGG19 & Naive Bayes)                | 0.48     | 0.59      | 0.48   | 0.48     |
| Hybrid KNN-SVM-Logistic-Regression Stacked Model  | 0.71     | 0.71      | 0.71   | 0.71     |
| Hybrid VGG16-SVM-Logistic-Regression Voting Model | 0.84     | 0.86      | 0.84   | 0.84     |
| Hybrid ResNet50-SVM-Naive-Bayes Stacked Model     | 0.91     | 0.92      | 0.91   | 0.91     |
| Hybrid ResNet50-KNN-Naive-Bayes Stacked Model     | 0.76     | 0.76      | 0.76   | 0.76     |
| Hybrid Model (SVM $ Logistic Regression) Voting   | 0.64     | 0.64      | 0.63   | 0.64     |
| Hybrid Model KNN-Logistic-Regression-Naive-Bayes  | 0.66     | 0.66      | 0.66   | 0.66     |

Some of the models above overfit to the training data, and performed not as well on the test data.

## Analysis of Image Classification Models

### Overview of Results

> > **Best Performing Model**

The CNN (ResNet50) significantly outperforms all other models with an
accuracy of 0.98 across all metrics. This suggests that deep learning approaches are particularly
well-suited for this image classification task.

> > **Traditional Machine Learning Models**

Among the basic
ML algorithms, Support Vector Machine (SVM) performs the best with 0.68 accuracy, followed closely by
Logistic Regression at 0.66. Naive Bayes and K-Nearest Neighbors show lower performance.

> > **Hybrid Models**

The VGG16 & Logistic Regression hybrid shows excellent performance (0.91 across all metrics), second
only to the pure CNN approach. VGG16 & SVM also performs well with 0.84 accuracy

Surprisingly, the ResNet50 & KNN hybrid (0.70 accuracy) doesn't perform as well as pure ResNet50, suggesting that the KNN classifier might be limiting the model's potential.
The VGG19 & Naive Bayes hybrid shows poor performance (0.48 accuracy), indicating this combination might not be suitable for the task.

> > **Ensemble Model**

The stacked KNN-SVM-Logistic-Regression model shows moderate performance (0.71 accuracy), slightly
better than individual traditional ML models but not as good as the best hybrid models.

## Key Takeaways

- **Deep Learning Superiority:** The CNN model demonstrates the power of deep learning in image
  classification tasks, significantly outperforming traditional ML approaches.
- **Effective Feature Extraction:** The success of hybrid models (especially VGG16 with Logistic
  Regression or SVM) indicates that these pre-trained networks are excellent feature extractors for this
  dataset.
- **Hybrid Model Variability:** The performance of hybrid models varies greatly, from very good
  (VGG16 & Logistic Regression) to poor (VGG19 & Naive Bayes), highlighting the importance of choosing
  the right combination of feature extractor and classifier.
- **Traditional ML Limitations:** While simpler to implement, traditional ML models
  (Logistic Regression, KNN, SVM, Naive Bayes) show limited performance in this task compared to deep
  learning approaches.
- **Ensemble Methods:** The stacked ensemble model shows some improvement over individual traditional
  ML models, but doesn't match the performance of the best hybrid or pure CNN models.
