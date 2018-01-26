import cv2
import glob2

def detectface(image):
    '''Function to Detect faces from given image file and print them inside rectangles '''
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img=cv2.imread(image)
    img_g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces=face_cascade.detectMultiScale(img_g,
        scaleFactor=1.1,
        minNeighbors=10)

    for x,y,w,h in faces:
        img=cv2.rectangle(img,(x,y),(x+w,y+h),(200,100,50),3)
    
    imgr=cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
    cv2.imshow(image+'_faces',imgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    images=glob2.glob('Images\*.jpg')
    for image in images:
        detectface(image)
        
if __name__=='__main__':
    main()