
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


from PIL import Image
from numpy import array
img = Image.open("circle/0.png")
arr = array(img)


# In[3]:


shape_name = ["Circle","Square","Star","Triangle"]


# In[4]:


shapes = []
labels = []
for i in range(0,3720): #3720
    img = Image.open("circle/" + str(i) + ".png")
    img=img.resize((30,30))
    arr = array(img)
    shapes.append(arr)
    labels.append([1,0,0,0])


# In[5]:


for i in range(0,3765): #3765
    img = Image.open("square/" + str(i) + ".png")
    img=img.resize((30,30))
    arr = array(img)
    shapes.append(arr)
    labels.append([0,1,0,0])


# In[6]:


for i in range(0,3765): #3765
    img = Image.open("star/" + str(i) + ".png")
    img=img.resize((30,30))
    arr = array(img)
    shapes.append(arr)
    labels.append([0,0,1,0])


# In[7]:


for i in range(0,3720):  #3720
    img = Image.open("triangle/" + str(i) + ".png")
    img=img.resize((30,30))
    arr = array(img)
    shapes.append(arr)
    labels.append([0,0,0,1])


# In[8]:


for i in range(len(shapes)):
    shapes[i] = shapes[i].flatten()


# In[9]:


shapes = np.asarray(shapes)
labels = np.asarray(labels)


# In[10]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[11]:


from sklearn.cross_validation import train_test_split


# In[12]:


X_us_train, X_us_test, y_us_train, y_us_test = train_test_split(shapes, labels, test_size=0.25, random_state=42)
scaler.fit(X_us_train)
X_train = scaler.transform(X_us_train)
scaler.fit(X_us_test)
X_test = scaler.transform(X_us_test)
scaler.fit(y_us_train)
y_train = scaler.transform(y_us_train)
scaler.fit(y_us_test)
y_test = scaler.transform(y_us_test)


# In[13]:


def Display_shape_test(num):
    #print(y_test[num])
    label=y_test[num].argmax(axis=0)
    image=X_us_test[num].reshape([30,30])
    #plt.title('Example: %d Label: %d'%(num,shape_name[label]))
    plt.imshow(image,cmap=plt.get_cmap('gray_r'))
    plt.show()


# In[14]:


def Display_shape_train(num):
    #print(y_train[num])
    label=y_train[num].argmax(axis=0)
    image=X__us_train[num].reshape([30,30])
    #plt.title('Example: %d Label: %d'%(num,shape_name[label]))
    plt.imshow(image,cmap=plt.get_cmap('gray_r'))
    plt.show()


# In[15]:


tf.device("/gpu:0")


# In[16]:


sess = tf.Session()


# In[17]:


x = tf.placeholder(tf.float32,[None,900])
y_ = tf.placeholder(tf.float32,[None,4])
W = tf.Variable(tf.zeros([900,4]))
b = tf.Variable(tf.zeros([4]))


# In[18]:


y = tf.nn.softmax(tf.matmul(x,W)+b)


# In[19]:


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
LEARNING_RATE = 0.1
TRAIN_STEPS = 2500


# In[20]:


sess.run(tf.global_variables_initializer())


# In[21]:


training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
for i in range(TRAIN_STEPS+1):
    sess.run(training,feed_dict={x:X_train,y_:y_train})
    if i%100 == 0:
        print('Training Step:' + str(i) + ' Accuracy = '+ str(sess.run(accuracy,feed_dict={x:X_test,y_:y_test})) + ' Loss = ' + str(sess.run(cross_entropy,{x:X_train,y_:y_train})))


# In[22]:


for i in range(0,11):
    Display_shape_test(i)
    answer = sess.run(y, feed_dict={x: X_test})

    print(shape_name[answer[i].argmax()])


# In[23]:


sess.close()

