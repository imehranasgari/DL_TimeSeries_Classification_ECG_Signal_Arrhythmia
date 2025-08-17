Of course\! I have enhanced and structured the information you provided by integrating the technical details from the notebook. The result is a comprehensive, professional README file optimized for your portfolio.

-----

# Heartbeat Classification Using Deep Learning on ECG Time Series Data

As a machine learning engineer passionate about applying AI to healthcare challenges, I developed this project to classify electrocardiogram (ECG) signals for arrhythmia detection. By implementing and rigorously comparing multiple deep learning architectures—including ANNs, CNNs, LSTMs, and a hybrid Transformer—I demonstrate my expertise in end-to-end time series classification. This work highlights my practical skills in handling imbalanced datasets, building complex models with TensorFlow and Keras, and leveraging ensemble techniques for improved performance, making it a strong addition to my portfolio for roles in AI-driven medical diagnostics.

-----

## Problem Statement and Goal of Project

The early detection of cardiac arrhythmias through ECG analysis is critical for patient care. However, manual interpretation is time-consuming and requires expert knowledge. This project addresses the challenge of automatically classifying ECG signals into different heartbeat categories, including normal beats and various types of anomalies.

A key technical challenge is the significant **class imbalance** in the dataset. For instance, the majority class (normal beats) contains **72,470 samples**, while the rarest class has only **641 samples**. The primary goal is to develop and evaluate robust classification models that can effectively learn from this imbalanced data and generalize well to unseen signals, providing a reliable foundation for automated arrhythmia detection systems.

-----

## Solution Approach

My approach involved a systematic, end-to-end machine learning workflow, from data preparation to model comparison and ensembling.

### 1\. Data Preprocessing and Feature Engineering

  * **Signal Filtering:** Applied notch and bandpass filters using `SciPy` to remove baseline wander and high-frequency noise from the raw ECG signals, ensuring the models trained on clean, relevant data.
  * **Data Balancing:** Addressed the severe class imbalance by implementing a hybrid resampling strategy using `scikit-learn`. I **upsampled** all minority classes to 20,000 samples and **downsampled** the majority class to 20,000 samples, creating a balanced dataset for training.
  * **Normalization:** Scaled the ECG signal amplitudes to a [0, 1] range to stabilize the training process.

### 2\. Model Implementation

I built four distinct deep learning architectures to compare their effectiveness:

  * **Artificial Neural Network (ANN):** A baseline `Dense` network to establish a performance benchmark.
  * **1D Convolutional Neural Network (CNN):** An architecture with `Conv1D`, `BatchNormalization`, and `MaxPooling1D` layers, designed to automatically extract hierarchical features from the ECG time series.
  * **Recurrent Neural Network (RNN):** An `LSTM`-based model to capture long-range temporal dependencies and patterns within the heartbeat signals.
  * **Hybrid Transformer Model:** A sophisticated hybrid model combining a CNN front-end for initial feature extraction with a Transformer encoder block to model complex relationships across the entire time series.

### 3\. Ensemble Methods

To further improve performance and robustness, I created two types of ensembles:

  * **Weighted Averaging:** Combined the predictions of the four deep learning models using optimized weights.
  * **Voting Classifier:** Integrated the deep learning models with traditional machine learning classifiers (`RandomForestClassifier`, `DecisionTreeClassifier`, `SVM`) in a voting ensemble.

### 4\. Evaluation

Each model was rigorously evaluated on the held-out test set using standard classification metrics: **Accuracy, Precision, Recall, and F1-Score (micro-averaged)**. I also generated confusion matrices to analyze class-specific performance in detail.

-----

## Technologies & Libraries

  * **Programming Language:** Python
  * **Deep Learning Frameworks:** TensorFlow, Keras
  * **Machine Learning Libraries:** scikit-learn (for resampling, metrics, and ensemble models)
  * **Time Series Processing:** Darts, SciPy (for signal filtering)
  * **Data Handling & Visualization:** NumPy, Pandas, Matplotlib, Seaborn

-----

## Description about Dataset

The project utilizes the **MIT-BIH Arrhythmia Database**, a widely recognized benchmark for heartbeat classification. The data is provided in two files:

  * `mitbih_train.csv`: **87,553 samples** for training.
  * `mitbih_test.csv`: **21,892 samples** for testing.

Each sample is an ECG signal represented by **187 time steps (features)** and is categorized into one of five classes:

  * **0:** Normal beat (N)
  * **1:** Supraventricular premature beat (S)
  * **2:** Premature ventricular contraction (V)
  * **3:** Fusion of ventricular and normal beat (F)
  * **4:** Unclassifiable beat (Q)

The training data exhibits a significant class imbalance, with the distribution as follows:

  * **Class 0 (Normal):** 72,470 samples
  * **Class 4 (Unclassifiable):** 6,431 samples
  * **Class 2 (PVC):** 5,788 samples
  * **Class 1 (SPB):** 2,223 samples
  * **Class 3 (Fusion):** 641 samples

-----

## Installation & Execution Guide

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2.  **Set up Environment:** Create and activate a Python virtual environment.

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *(Note: A `requirements.txt` file should be created containing `tensorflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `darts`, and `scipy`.)*

4.  **Download Dataset:** The notebook includes a command to download the dataset directly from KaggleHub.

5.  **Run Notebook:** Open and run the `class_timeseris.ipynb` notebook in a Jupyter environment.

-----

## Key Results / Performance

All models achieved high performance on the test set, demonstrating the effectiveness of the chosen architectures and preprocessing techniques. The ANN and Hybrid Transformer models were the top-performing individual models, while the ensembles provided comparable and robust results.

| Model               | Accuracy | Precision (Micro) | Recall (Micro) | F1-Score (Micro) |
| ------------------- | :------: | :---------------: | :------------: | :--------------: |
| ANN                 |  \~98.0%  |      \~98.0%       |     \~98.0%     |      \~98.0%      |
| CNN                 |  \~97.0%  |      \~97.0%       |     \~97.0%     |      \~97.0%      |
| RNN (LSTM)          |  \~96.0%  |      \~96.0%       |     \~96.0%     |      \~96.0%      |
| Hybrid Transformer  |  \~98.0%  |      \~98.0%       |     \~98.0%     |      \~98.0%      |
| Weighted Ensemble   |  \~98.0%  |        —          |       —        |        —         |
| Voting Ensemble     |  \~98.0%  |        —          |       —        |        —         |

-----

## Screenshots / Sample Output

**1. Sample ECG Signals by Class:**
This plot showcases the distinct waveform morphologies for each of the five heartbeat classes present in the dataset.

**2. Model Performance Comparison:**
The bar chart below provides a clear visual comparison of the evaluation metrics across the four deep learning models.

**3. Confusion Matrix for the Hybrid Transformer Model:**
This matrix visualizes the model's predictions, highlighting its strong performance in distinguishing between different arrhythmia types.

-----

## Additional Learnings / Reflections

This project was a valuable exercise in applying a diverse set of deep learning techniques to a critical healthcare problem. Key takeaways include:

  * **End-to-End Implementation:** I successfully managed the entire project lifecycle, from signal preprocessing and data balancing to building, training, and evaluating multiple complex neural networks.
  * **Handling Class Imbalance:** I gained hands-on experience with resampling techniques, which were crucial for preventing model bias and achieving high accuracy on minority classes.
  * **Architectural Trade-offs:** By implementing and comparing four different architectures, I developed a deeper intuition for their respective strengths—the feature extraction power of CNNs, the sequence awareness of LSTMs, and the global attention mechanism of Transformers.
  * **Ensemble Methods:** Building weighted and voting ensembles demonstrated my ability to combine model strengths for improved generalization and robustness, a key skill in production-level machine learning.

-----

💡 *Some interactive outputs (e.g., plots, widgets) may not display correctly on GitHub. If so, please view this notebook via [nbviewer.org](https://nbviewer.org) for full rendering.*

## 👤 Author

## Mehran Asgari

## **Email:** [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com).

## **GitHub:** [https://github.com/imehranasgari](https://github.com/imehranasgari).

-----

## 📄 License

This project is licensed under the Apache 2.0 License – see the `LICENSE` file for details.