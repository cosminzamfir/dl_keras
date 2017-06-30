import random
import string
import cv2

text = ''
text = text.join([random.choice(string.ascii_letters) for i in range(10)])

print(text)

y0, dy = 50, 4
for i, line in enumerate(text.split('\n')):
    y = y0 + i*dy
    cv2.putText(img, line, (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)