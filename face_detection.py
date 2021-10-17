# import opencv python library
import cv2

# load trained XML classifier file using Cascadeclassifier()
detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Capture frames from Camera or Image using videocapture()
imp_img = cv2.VideoCapture('ratan_tata.jpg')

# After capturing, read the image using read()
# Reading image gives two returns: result(True/False), Pixel Coordinates of image
res, img = imp_img.read()

# Convert image into gray scale because haar cascade classifer is trained for gray scale images using cvtcolor()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Now Detect faces in image using detectMultiscale()
faces = detect.detectMultiScale(gray, 1.3, 5)

# Use for loop to iterate faces and use x,y,w,h to draw square
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 3)

# Show image
cv2.imshow("Ratan Tata", img)

# Duration of image to display (msecs)
cv2.waitKey(5000)

# close the window
imp_img.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
