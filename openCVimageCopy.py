import cv2

print("OpenCV version:")
print(cv2.__version__)
 
img = cv2.imread("./image/nomadProgramerIcon.png")
print("width: {} pixels".format(img.shape[1]))
print("height: {} pixels".format(img.shape[0]))
print("channels: {}".format(img.shape[2]))
 
cv2.imshow("nomadProgramer", img)
 
cv2.waitKey(0)
cv2.imwrite("./image/nomadProgramerIcon.jpg", img)
cv2.destroyAllWindows()
