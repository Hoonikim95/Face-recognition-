import cv2
import face_recognition
import pickle
import time

file_name = './video/tedy_01.mp4'
encoding_file = './model/encodings.pickle'
unknown_name = 'Unknown'
output_name = './video/output_' + model_method + '.avi'
frame_count = 0
recognition_count = 0
elapsed_time = 0
# CNN method is more accurate but slower. HOG is faster but less accurate.
model_method = 'hog'

def detectAndDisplay(image):
    start_time = time.time()
    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb,
        model=model_method)
    encodings = face_recognition.face_encodings(rgb, boxes)

    names = []

    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"],
            encoding)
        name = unknown_name

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
            global recognition_count
            recognition_count += 1
        
        names.append(name)

    for ((top, right, bottom, left), name) in zip(boxes, names):
        y = top - 15 if top - 15 > 15 else top + 15
        color = (0, 255, 0)
        line = 2
        if(name == unknown_name):
            color = (0, 0, 255)
            line = 1
            name = ''
            
        cv2.rectangle(image, (left, top), (right, bottom), color, line)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.75, color, line)
        
    frame_time = time.time() - start_time
    global elapsed_time
    elapsed_time += frame_time
    print("Frame {} time {:.3f} seconds".format(frame_count, frame_time))
 
    cv2.imshow("Recognition", image)
    
    global writer
    if writer is None and output_name is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_name, fourcc, 24,
                (image.shape[1], image.shape[0]), True)

    if writer is not None:
        writer.write(image)
		
data = pickle.loads(open(encoding_file, "rb").read())

vs = cv2.VideoCapture(file_name)
writer = None
if not vs.isOpened:
    print('### Error opening video ###')
    exit(0)
while True:
    ret, frame = vs.read()
    frame_count += 1
    if frame is None:
        print('### No more frame ###')
        vs.release()
        writer.release()
        break
    detectAndDisplay(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Recognition {} per {}, Elapsed time {:.3f} seconds".format(recognition_count, frame_count, elapsed_time))
cv2.destroyAllWindows()

