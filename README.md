# OCR-Smart-Camera

This script provides a user-friendly interface for performing optical character recognition (OCR) on images, using Tesseract, a open source OCR engine. Users can select an image file either from their computer or capture it through a webcam. The selected image then goes through 9 preprocessing stpes using a computer vision librabry called OpenCV. The text present in the image is extracted using the Tesseract OCR engine.

Once the text is extracted, users can choose a target language from a dropdown menu. The extracted text is then translated into the selected language using the Google Translate API. The translated text is displayed in the GUI, allowing users to read and review the translation.

Additionally, users have the option to play the translated text as audio. The script converts the translated text into speech using the gTTS library and saves it as an audio file. The audio file is then played using the system's default media player, allowing users to listen to the translated text.

Overall, this script combines OCR, translation, and text-to-speech functionalities in a single application, making it convenient for users to extract, translate, and listen to the text present in images.


This app can be used for extracting text from images, translating the extracted text to different languages, and playing the translated text as audio. It provides a convenient way to process and understand text content in images, enabling tasks such as multilingual document analysis, language translation, and audio-based language learning.


Dependencies

OpenCV (cv2)
NumPy (numpy)
Pandas (pandas)
scikit-learn (sklearn)
PyTesseract (pytesseract)
Googletrans (googletrans)
gTTS (gtts)
Tkinter (tkinter)
network error
