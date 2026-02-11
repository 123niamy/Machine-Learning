# SMS Spam Detection (Machine Learning Portfolio Project)

This project demonstrates practical machine learning skills by building a model to classify SMS messages as 'spam' or 'ham' (legitimate).

## Project Overview
- Uses the real SMSSpamCollection dataset
- Preprocesses SMS text and applies TF-IDF vectorization
- Trains a Naive Bayes classifier (MultinomialNB)
- Evaluates model performance (accuracy, precision, recall)
- Provides a function to predict new SMS messages

## Skills Demonstrated
- Data loading and cleaning
- Text preprocessing
- Feature extraction (TF-IDF)
- Model training and validation
- Performance reporting
- Python scripting and scikit-learn usage

## Usage
1. Install required libraries:
   - pandas
   - scikit-learn
   - numpy
2. Run `sms_spam_detection.py` to train and evaluate the model.
3. Use the `predict_sms()` function to classify new SMS messages.

### Example
```python
result = predict_sms("Congratulations! You've won a free ticket. Reply now!")
print(result)  # Output: 'spam' or 'ham'
```

## Portfolio Value
This project is suitable for showcasing machine learning skills to employers and collaborators. It can be extended with more advanced models, web interfaces, or additional datasets.

## Author
- GitHub: [123niamy](https://github.com/123niamy/Machine-Learning)

## License
This project is for educational and portfolio purposes.
