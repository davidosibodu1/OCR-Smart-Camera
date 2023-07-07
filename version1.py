import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pytesseract
from googletrans import Translator
from gtts import gTTS
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import filedialog

#df = pd.read_csv('ocr_dataset.csv')

# Split the dataset
#train = df[:800]
#test = df[800:]

# Extract features
#def extract_features(img):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #edges = cv2.Canny(gray, 100, 200)
    #contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #areas = [cv2.contourArea(cnt) for cnt in contours]
    #return np.array([np.mean(areas), np.std(areas)])

# Extract features from training set
#train_features = np.array([extract_features(cv2.imread(file)) for file in train['path']])
#train_labels = train['label']

# Train the KNN algorithm
#knn = KNeighborsClassifier(n_neighbors=3)
#knn.fit(train_features, train_labels)


root = tk.Tk()
root.title("Tesseract OCR text translator")

image_label = tk.Label(root)
image_label.grid(row=6, column=0, columnspan=3)

tk.Label(root, text="Image: ").grid(row=0, column=0)
input_path = tk.Entry(root)
input_path.grid(row=0, column=1)


tk.Label(root, text="Language Codes:", font=("Arial", 12)).grid(row=0, column=10, sticky="w")

tk.Label(root, text="1.|eng/en: English", font=("Arial", 12)).grid(row=1, column=10, sticky="w")

tk.Label(root, text="2.|fra/fr: French", font=("Arial", 12)).grid(row=2, column=10, sticky="w")
 
tk.Label(root, text="3.|deu/de: German", font=("Arial", 12)).grid(row=3, column=10, sticky="w")

tk.Label(root, text="4.|ar: Arabic", font=("Arial", 12)).grid(row=1, column=11, sticky="w")

tk.Label(root, text="5.|nl: Dutch", font=("Arial", 12)).grid(row=2, column=11, sticky="w")

tk.Label(root, text="6.|por/po: Portugal", font=("Arial", 12)).grid(row=3, column=11, sticky="w")

tk.Label(root, text="7.|ru: Russian", font=("Arial", 12)).grid(row=1, column=12, sticky="w")

tk.Label(root, text="8.|ko: Korean", font=("Arial", 12)).grid(row=2, column=12, sticky="w")

tk.Label(root, text="9.|ja: Japanese", font=("Arial", 12)).grid(row=3, column=12, sticky="w")

tk.Label(root, text="10.|yo: Yoruba", font=("Arial", 12)).grid(row=1, column=13, sticky="w")

tk.Label(root, text="11.|hi: Hindi", font=("Arial", 12)).grid(row=2, column=13, sticky="w")

tk.Label(root, text="12.|es/spa: Spanish", font=("Arial", 12)).grid(row=3, column=13, sticky="w")

tk.Label(root, text="13.|zh-CN: Chinese(Simplfied)", font=("Arial", 12)).grid(row=1, column=14, sticky="w")

tk.Label(root, text="14.|bn: bengali", font=("Arial", 12)).grid(row=2, column=14, sticky="w")

tk.Label(root, text="15.|ur: Urdu", font=("Arial", 12)).grid(row=3, column=14, sticky="w")

def select_image():
    path = tk.filedialog.askopenfilename()
    input_path.delete(0, tk.END)
    input_path.insert(0, path)
    show_image(path)
    
    
def show_image(path):
    global photo
    image = Image.open(path)
    image = image.resize((800, 800), Image.LANCZOS)
    photo = ImageTk.PhotoImage(image)
    image_label.configure(image=photo)
    
def show_camera_feed():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Camera Preview")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Camera Preview", frame)
        if cv2.waitKey(1) == ord('q'):
            cv2.imwrite("captured_image.jpg", frame)
            break
        if cv2.waitKey(1) == ord('f'):
            
            cap.set(cv2.CAP_PROP_FOCUS, 0)
            print("Focus set to infinity")
    cap.release()
    cv2.destroyAllWindows()
    
    
select_button = tk.Button(root, text="Select a picture", command=select_image)
select_button.grid(row=0, column=2)

preview_button = tk.Button(root, text="Preview Camera:press q to take an image", command=show_camera_feed)
preview_button.grid(row=0, column=3)

tk.Label(root, text="Original language of text: ").grid(row=1, column=0)
input_lang = ttk.Combobox(root, values=['eng', 'fra', 'spa', 'deu', 'ar', 'nl', 'it', 'por', 'ru', 'ko', 'ja', 'yo', 'hi', 'zh-CN', 'bn', 'ur'])
input_lang.grid(row=1, column=1)
input_lang.current(0)


output_languages = {
    "en": "en",
    "fr": "fr",
    "es": "es",
    "de": "de",
    "ar": "ar",
    "nl": "nl",
    "it": "it",
    "pt": "pt",
    "ru": "ru",
    "ko": "ko",
    "ja": "ja",
    "yo": "yo",
    "hi": "hi",
    "zh-CN": "zh-CN",
    "bn": "bn",
    "ur": "ur"
    }
    

tk.Label(root, text="Translate Language To: ").grid(row=2, column=0)
output_lang = ttk.Combobox(root, values=list(output_languages.values()))
output_lang.grid(row=2, column=1)
output_lang.current(0)


def extract_text():
    # Load the image
    image = cv2.imread(input_path.get())
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # Convert the image to grayscale
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    alpha = 1.5
    beta = 0
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    blurred = cv2.GaussianBlur(adjusted, (5, 5), 0)
    
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Perform erosion to remove noise
    erode = cv2.erode(binary, None, iterations=1)

    # Perform dilation to fill in gaps in the text
    dilate = cv2.dilate(erode, None, iterations=1)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    

    # Extract text from the image using Tesseract OCR
    text = pytesseract.image_to_string(closing, lang=input_lang.get())

    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, text)


def translate_text():
    translator = Translator()
    text = output_text.get("1.0", tk.END)
    translated_text = translator.translate(text, dest=output_lang.get()).text
    translated_output.delete("1.0", tk.END)
    translated_output.insert(tk.END, translated_text)


def play_audio():
    text = translated_output.get("1.0", tk.END)
    tts = gTTS(text, lang=output_lang.get())
    tts.save("translation.mp3")
    os.system("mpg321 translation.mp3")


tk.Button(root, text="Extract Text", command=extract_text).grid(row=3, column=0)
tk.Button(root, text="Translate", command=translate_text).grid(row=3, column=1)
tk.Button(root, text="Play Audio translation", command=play_audio).grid(row=3, column=2)

# Print the extracted text
output_text = tk.Text(root, height=10, width=50)
output_text.grid(row=4, column=0, columnspan=3)

translated_output = tk.Text(root, height=10, width=50)
translated_output.grid(row=5, column=0, columnspan=3)

root.mainloop()