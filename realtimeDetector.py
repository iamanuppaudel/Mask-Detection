#%%
import cv2
import numpy as np
import tensorflow as tf  
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model= load_model("Mask_Detection_CNN_1596783544")
CATEGORIES=['WithoutMask','WithMask']
#for testing
def test():
    test_img= image.load_img('temp.png', target_size=(80,80))
    test_img= image.img_to_array(test_img)
    test_img= np.expand_dims(test_img, axis=0)
    prediction= model.predict(test_img)
    finalPrediction=CATEGORIES[int(prediction[0][0])]
    print(CATEGORIES[int(prediction[0][0])])
    return finalPrediction



cam=cv2.VideoCapture(0)

cv2.namedWindow("test")
img_text=''

while True:
    ret, frame = cam.read()
    frame= cv2.flip(frame,1)
    img= cv2.rectangle(frame, (100,50),(360,430),(0,0,255), thickness=2 )
    imcrop=img[51:429, 101:359]

    cv2.putText(frame, img_text,(10,470),cv2.FONT_HERSHEY_TRIPLEX, 1.5,(0,0,255))
    cv2.imshow("test", frame)

    img_name="temp.png"
    #save_img= cv2.resize(frame)
    cv2.imwrite(img_name,imcrop)
    img_text= test()

    if cv2.waitKey(1) & 0xFF== ord('q'):
        break

cam.release()
cv2.destroyAllWindows()






# %%
