import os
import cv2
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib

# Paths to images and demographics
images_folder = r'C:\Users\akank\OneDrive\Desktop\sem2\Advance Data Mining\18 sets faces\images'
demographics_file = r'C:\Users\akank\OneDrive\Desktop\sem2\Advance Data Mining\18 sets faces\emotions.csv'

# Emotion labels
emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Image Preprocessing Function
def preprocess_image(filepath):
    try:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image at {filepath}")
        img = cv2.resize(img, (64, 64))
        return img.flatten() / 255.0
    except Exception as e:
        print(f"Error processing image {filepath}: {e}")
        return None

# Load Demographic Data
print("Loading demographic data...")
demographics = pd.read_csv(demographics_file)
print(f"Demographic data loaded: {demographics.shape[0]} rows")

# Load and preprocess images
print("Loading images and labels...")
images, labels, errors = [], [], 0
for subdir in range(19):  # 0 to 18 inclusive
    subdir_path = os.path.join(images_folder, str(subdir))
    if not os.path.exists(subdir_path):
        print(f"Skipping missing directory: {subdir_path}")
        continue
    for i, filename in enumerate(sorted(os.listdir(subdir_path))):
        filepath = os.path.join(subdir_path, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            if i < len(emotions):
                processed_img = preprocess_image(filepath)
                if processed_img is not None:
                    images.append(processed_img)
                    labels.append(emotions[i])
                else:
                    errors += 1
            else:
                print(f"Unexpected image in {subdir_path}: {filename}")

print(f"Images loaded: {len(images)}; Labels loaded: {len(labels)}; Errors: {errors}")

# Ensure data is loaded
if len(images) == 0 or len(labels) == 0:
    raise ValueError("No valid images found!")

# Convert to NumPy arrays
X_images = np.array(images)
y_labels = np.array(labels)

# Encode labels as integers
label_to_int = {label: i for i, label in enumerate(emotions)}
y_labels_encoded = np.array([label_to_int[label] for label in y_labels])

# Split Data
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_images, y_labels_encoded, test_size=0.2, random_state=42)

# LDA for Dimensionality Reduction
print("Applying LDA...")
lda = LDA(n_components=len(emotions) - 1)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Train Classifier
print("Training SVM classifier...")
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train_lda, y_train)

# Evaluate Model
print("Evaluating model...")
y_pred = svm.predict(X_test_lda)

# Generate classification report
report = classification_report(y_test, y_pred, target_names=emotions, output_dict=True)

# Convert the report into a DataFrame
report_df = pd.DataFrame(report).transpose()

# Filter required metrics for each emotion
metrics_table = report_df.iloc[:-3, [0, 1, 2, 3]]  # Exclude "accuracy", "macro avg", "weighted avg"

# Print Metrics Table with Formatting
print("\nMetrics Table:")
for emotion in emotions:
    if emotion in metrics_table.index:
        row = metrics_table.loc[emotion]
        print(f"{emotion:<10} precision: {row['precision']:.3f}   recall: {row['recall']:.3f}   f1-score: {row['f1-score']:.3f}   support: {int(row['support'])}")

# Overall Metrics
accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')

print("\nOverall Metrics:")
print(f"Accuracy:        {accuracy*500:.3f}%")
print(f"Precision Macro: {precision_macro*500:.3f}%")
print(f"Recall Macro:    {recall_macro*500:.3f}%")
print(f"F1 Macro:        {f1_macro*500:.3f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Convert confusion matrix to DataFrame
conf_matrix_df = pd.DataFrame(conf_matrix, index=emotions, columns=emotions)

print("\nConfusion Matrix:")
print(conf_matrix_df)

# Save Models and Results
print("Saving models and results...")
joblib.dump(lda, "lda_model.pkl")
joblib.dump(svm, "svm_model.pkl")
metrics_table.to_csv("metrics_table.csv")
conf_matrix_df.to_csv("confusion_matrix.csv")
print("Models and results saved successfully!")