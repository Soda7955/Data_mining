import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

# Load data
data = pd.read_csv('train_weka2.csv')

# Data preprocessing
data['text'] = data['context'] + " " + data['response']
label_mapping = {
    "__needs_caution__": 0,
    "__casual__": 1,
    "__needs_intervention__": 2,
    "__possibly_needs_caution__": 3,
    "__probably_needs_caution__": 4
}
data['safety_label'] = data['safety_label'].map(label_mapping)
X = data['text']
y = data['safety_label']

# Text vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Data standardization
scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training: Logistic Regression
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Model evaluation
predictions = model.predict(X_test_scaled)
print(classification_report(y_test, predictions))

# Try using Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)
print(classification_report(y_test, rf_predictions))

# Handle class imbalance using SMOTE
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
model.fit(X_train_smote, y_train_smote)
smote_predictions = model.predict(X_test_scaled)
print(classification_report(y_test, smote_predictions))

# Predict new texts
new_texts = ["I want to hurt you!", "Everything is wonderful!"]
new_texts_vectorized = vectorizer.transform(new_texts)
new_texts_scaled = scaler.transform(new_texts_vectorized)
new_predictions = model.predict(new_texts_scaled)
for text, prediction in zip(new_texts, new_predictions):
    print(f"The predicted safety label for '{text}' is: {prediction}")

