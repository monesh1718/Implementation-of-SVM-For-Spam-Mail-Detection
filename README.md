# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: monesh s
RegisterNumber:  35006689
*/


# Import necessary libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the data

df = pd.read_csv('spam.csv', encoding='latin-1')

# Keep only the first two columns and rename them

df = df.iloc[:, :2]
df.columns = ['label', 'message']

# Convert labels(ham = 0, spam = 1) :

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Remove any rows with missing values

df = df.dropna()

# Display basic info

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nClass distribution:")
print(df['label'].value_counts())

# Split data into features (X) and target (y)

X = df['message']
y = df['label']

# Split into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numerical features using TF-IDF

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Create and train SVM model

svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_vectorized, y_train)

# Make predictions

y_pred = svm_model.predict(X_test_vectorized)

# Evaluate the model

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Test with some example messages

test_messages = [
    "Hey, are we still meeting for lunch tomorrow?",
    "CONGRATULATIONS! You've won a FREE cruise to the Andaman! Call now to claim your prize!",
    "Can you pick up some milk on your way home?",
    "URGENT! Your account has been suspended. Click here to verify your details immediately."
]

print("\n" + "="*50)
print("TESTING WITH EXAMPLE MESSAGES:")
print("="*50)

for msg in test_messages:
    msg_vectorized = vectorizer.transform([msg])
    prediction = svm_model.predict(msg_vectorized)[0]
    result = "SPAM" if prediction == 1 else "HAM"
    print(f"Message: {msg[:50]}... -> {result}")

```

## Output:
<img width="613" height="202" alt="564275300-f02ff148-bb3b-4d9a-a868-552c95d7774e" src="https://github.com/user-attachments/assets/1a8922a9-0781-472e-8cd1-f361082b64db" />

<img width="263" height="145" alt="564275492-f9b5cf73-08b3-464c-a44b-540b7e7dc949" src="https://github.com/user-attachments/assets/5d7652e2-ebe6-4870-b649-7359a8d61791" />
<img width="545" height="197" alt="564275845-37522042-faa0-4a63-849a-161dbe559543" src="https://github.com/user-attachments/assets/dbc8eaf1-c796-493c-8201-991fa4a41d28" />
<img width="692" height="134" alt="564276002-f292f586-a021-46b5-8dff-3ab0f1854542" src="https://github.com/user-attachments/assets/a0a7bd31-f95d-4914-af59-f28caef027db" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
