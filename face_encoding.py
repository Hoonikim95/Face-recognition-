import cv2
import face_recognition
import pickle

dataset_paths = ['./image/tedy/']
names = ['Tedy']
number_images = 10
image_type = '.jpg'
encoding_file = './model/encodings.pickle'
# Either cnn  or hog. The CNN method is more accurate but slower. HOG is faster but less accurate.
model_method = 'cnn'

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, dataset_path) in enumerate(dataset_paths):
    # extract the person name from names
    name = names[i]

    for idx in range(number_images):
        file_name = dataset_path + str(idx+1) + image_type

        # get RGB image from BGR, OpenCV format
        image = cv2.imread(file_name)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect a coordinates of the bounding boxes
        boxes = face_recognition.face_locations(rgb,
            model=model_method)

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            # append each encoding + name to both array
            print(file_name, name, encoding)
            knownEncodings.append(encoding)
            knownNames.append(name)
        
# make the facial encodings file
data = {"encodings": knownEncodings, "names": knownNames}
f = open(encoding_file, "wb")
f.write(pickle.dumps(data))
f.close()
