import re
import cv2
import pytesseract
from pytesseract import Output

image = cv2.imread('./image/receipt_01.jpg')

# hsv transform - value = gray image
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hue, saturation, value = cv2.split(hsv)

d = pytesseract.image_to_data(value, output_type=Output.DICT)
keys = list(d.keys())
print(keys)

data_pattern = '^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/(19|20)?\d\d$'
data_pattern2 = "(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
# data_pattern =  'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'

n_boxes = len(d['text'])
print(n_boxes)
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
    	if re.match(data_pattern, d['text'][i]):
	        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
	        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	        cv2.putText(image, d['text'][i], (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	        print(d['text'][i])

for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
    	if re.match(data_pattern2, d['text'][i]):
	        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
	        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	        cv2.putText(image, d['text'][i], (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	        print(d['text'][i])

cv2.imshow('OCR', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
