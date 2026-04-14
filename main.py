import pytesseract
import cv2
def get_text(img):
    text = pytesseract.image_to_string(img)
    return text

img = cv2.imread("/home/gautamk/2026-04-11-090401_hyprshot.png")
print(get_text(img))