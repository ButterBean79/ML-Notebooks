#!/usr/bin/env python
# coding: utf-8

# In[11]:


import cv2
import imutils


# In[12]:


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


# In[13]:


image = cv2.imread('./image.png')


# In[14]:


image = imutils.resize(image, width=min(400, image.shape[1]))


# In[15]:


(regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)


# In[16]:


for (x, y, w, h) in regions:
    cv2.rectangle(image, (x, y), (x + w, y + h), (25, 175, 255), 1)


# In[17]:


cv2.imshow("Image", image)
cv2.waitKey(0)

cv2.destroyAllWindows()

cv2.imwrite("./detected_image.png", image)


# In[18]:


cap = cv2.VideoCapture('video.mp4')
   
while cap.isOpened():
    ret, image = cap.read()
    if ret:
        image = imutils.resize(image, width=min(400, image.shape[1]))
   
        (regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.05)
   
        for (x, y, w, h) in regions:
            cv2.rectangle(image, (x, y), (x + w, y + h), (25, 178, 255), 2)
   
        cv2.imshow("Image", image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
  
cap.release()
cv2.destroyAllWindows()


# In[ ]:




