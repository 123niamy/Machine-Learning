# SMS Spam Detection

This project builds a machine learning model to classify SMS messages as 'spam' or 'ham' (legitimate).

## Features
- Uses the SMSSpamCollection dataset
- Preprocesses SMS text and applies TF-IDF vectorization
- Trains a Naive Bayes classifier (MultinomialNB)
- Evaluates model performance (accuracy, precision, recall)
- Provides a function to predict new SMS messages

## Usage
1. Ensure you have the required libraries installed:
   - pandas
   - scikit-learn
   - numpy
2. Place the SMSSpamCollection dataset in the correct path if you want to retrain.
3. Run the script to train and evaluate the model.
4. Use the `predict_sms()` function to classify new SMS messages.

### Example
```python
result = predict_sms("Congratulations! You've won a free ticket. Reply now!")
print(result)  # Output: 'spam' or 'ham'
```

## Author
- GitHub: [123niamy](https://github.com/123niamy/Machine-Learning)

## License
This project is for educational purposes.
# AI Engineering Portfolio

A comprehensive collection of AI and Data Engineering projects covering the full spectrum of modern AI development.

##  Course Structure

This repository is organized into 5 core areas of AI Engineering:

### 01 - Data Engineering
ETL pipelines, data validation, transformation, and storage solutions.
- **Iris Data Pipeline**: 7-stage comprehensive ETL with governance
- **Titanic Pipeline**: Data cleaning, feature engineering, and visualization
- **Simple CSV Ingestion**: Basic database storage patterns

### 02 - Machine Learning
Classical ML algorithms, model training, and evaluation.
- **Titanic Survival Prediction**: Logistic regression classification model

### 03 - Generative AI
Generative models and adversarial training.
- **GAN Iris Generator**: Generative Adversarial Network for synthetic data generation

### 04 - Natural Language Processing
Text processing, sentiment analysis, language understanding.
- *Coming soon*

### 05 - Computer Vision
Image classification, object detection, and visual recognition.
- *Coming soon*

##  Tech Stack

- **Languages**: Python 3.14
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Deep Learning**: PyTorch
- **Visualization**: matplotlib, seaborn
- **Databases**: SQLite3
- **Development**: Jupyter, VS Code

##  Repository Structure

```
AI-Engineering/
 01-data-engineering/
    exercises/          # Daily practice exercises
    [pipeline files]    # Production-ready ETL pipelines
 02-machine-learning/
    exercises/          # ML algorithm practice
    [model files]       # Trained models and experiments
 03-generative-ai/
    exercises/          # GenAI applications
    GAN_iris_generator.py
 04-natural-language-processing/
    exercises/          # NLP projects
 05-computer-vision/
     exercises/          # CV implementations
```

##  Usage

Each project includes:
- Clear documentation and comments
- Error handling and logging
- Reproducible results with fixed random seeds
- Visualization outputs

To run any pipeline:
```bash
cd [course-folder]
python [script-name].py
```

##  Learning Journey

This repository tracks my continuous learning in AI Engineering, with daily exercises and projects across multiple disciplines. Each course folder represents hands-on experience with real-world problems and production-ready code patterns.

##  License

This project is for educational and portfolio purposes.

---

**Author**: 123niamy  
**Last Updated**: December 2025
