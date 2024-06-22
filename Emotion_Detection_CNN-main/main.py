from keras.models import load_model#Alexnet
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
#here to below
# Load the model file
model_path = r'C:\Users\Maroof\Desktop\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\model.h5'
with h5py.File(model_path, 'r') as file:
    # Check model architecture
    print("Model Architecture:")
    model_summary = file.attrs['model_config']
    print(model_summary)

    # Extract training history
    history = {}
    if 'history' in file:
        for key in file['history'].keys():
            history[key] = file['history'][key][:]
if 'accuracy' in history and 'val_accuracy' in history:
# Plot training & validation accuracy values
    plt.plot(history['accuracy'], label='Train')
    plt.plot(history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()
#here to above
# Ensure the Haar cascade file path is correct
haarcascade_path = r'C:\Users\Maroof\Desktop\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml'
if not os.path.exists(haarcascade_path):
    raise FileNotFoundError(f"Haar cascade file not found: {haarcascade_path}")

face_classifier = cv2.CascadeClassifier(haarcascade_path)
classifier = load_model(r'C:\Users\Maroof\Desktop\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#----------------------------------

# from keras.models import load_model
# from keras.preprocessing.image import img_to_array
# import cv2
# import numpy as np
#
# face_classifier = cv2.CascadeClassifier(r'F:\ML MODELS\Emotion_Detection_CNN-main/haarcascade_frontalface_default.xml')
# classifier = load_model(r'C:\Users\Mazhar Hayat\Downloads\Emotion_Detection_CNN-main/model.h5')
#
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
#
# # Load image
# image_path = 'C:\\Users\\Mazhar Hayat\\Downloads\\Emotion_Detection_CNN-main\\images\\validation\sad\\22690.jpg'
#
# frame = cv2.imread(image_path)
#
# labels = []
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# faces = face_classifier.detectMultiScale(gray)
#
# for (x, y, w, h) in faces:
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
#
#     if np.sum([roi_gray]) != 0:
#         roi = roi_gray.astype('float') / 255.0
#         roi = img_to_array(roi)
#         roi = np.expand_dims(roi, axis=0)
#
#         prediction = classifier.predict(roi)[0]
#         label = emotion_labels[prediction.argmax()]
#         labels.append(label)
#     else:
#         labels.append('No Faces')
#
# print("Detected emotions:", labels)
#

#-------------------------------
#
# import cv2
# from time import sleep
# import keras.models
# from keras.preprocessing.image import img_to_array
# import numpy as np
#
# face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# classifier = keras.models.load_model('model.h5')
#
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
#
#
# # Function to classify emotions
# def classify_emotion(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#     labels = []
#     for (x, y, w, h) in faces:
#         roi_gray = gray[y:y + h, x:x + w]
#         roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
#
#         if np.sum([roi_gray]) != 0:
#             roi = roi_gray.astype('float') / 255.0
#             roi = img_to_array(roi)
#             roi = np.expand_dims(roi, axis=0)
#
#             prediction = classifier.predict(roi)[0]
#             label = emotion_labels[prediction.argmax()]
#             labels.append(label)
#         else:
#             labels.append('No Faces')
#
#     return labels
#
#
# # Main function
# def main():
#     cap = cv2.VideoCapture(0)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     total_frames = fps * 20  # 20 seconds
#     interval_frames = fps * 1  # 2 seconds
#
#     pictures = []
#     confident_count = 0
#     unconfident_count = 0
#
#     for i in range(total_frames):
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         if i % interval_frames == 0:
#             picture_labels = classify_emotion(frame)
#             pictures.append(picture_labels)
#
#             confident_count +=  picture_labels.count('Neutral')+ picture_labels.count('Disgust') + picture_labels.count('Surprise')+ +picture_labels.count('Happy')
#             unconfident_count += picture_labels.count('Fear') + picture_labels.count('Sad')+picture_labels.count('Angry')
#
#
#         cv2.imshow('Video', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#     total_pictures = len(pictures)
#     print("Total Pictures: ", total_pictures)
#     print("Confident: ", confident_count)
#     print("Unconfident: ", unconfident_count)
#     if confident_count > unconfident_count :
#         print("Result: Confident")
#     elif unconfident_count > confident_count :
#         print("Result: Unconfident")
#
#
# if __name__ == "__main__":
#     main()


