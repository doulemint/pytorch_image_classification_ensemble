import pandas as pd
#drop unknow artist
import matplotlib as mpl
import matplotlib.pyplot as plt
log_dir ='logs/'
mpl.rcParams['figure.figsize'] = (22, 20)
dataset=pd.read_csv('/content/MultitaskPainting100k_Dataset_groundtruth/groundtruth_multiloss_train_header.csv')
# indexName=pf[pf['artist']=='Unknown photographer'].index
# pf.drop(indexName,inplace=True)
# grouped = pf.groupby(['artist']).size().reset_index(name='counts')
# p=grouped.sort_values('counts', ascending=False).head(50)
# top50=p['artist'].tolist()
# dataset=pd.DataFrame()
# for name,group in pf.groupby(['artist']):
#   if name in top50:
#     dataset=pd.concat([dataset,group],axis=0)
# dataset=dataset.reset_index()

import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
def generate_classdict(label):
  counter = Counter(label)
  class_num=len(counter)
  class_list=list(counter.keys()) #?
  class_dict={}
  class_weight={}
  total = len(label)
  count=0
  for name,num in counter.items():
    class_dict[name]=count
    class_weight[count]=(total/(num*class_num))
    count+=1
  return class_num,class_list,class_dict,class_weight

X=np.array(dataset['filename'].tolist())
y=np.array(dataset['style'].tolist())
Style_class_num,Style_class_list,Style_class_dict,Style_class_weight=generate_classdict(y)
y=np.array(dataset['genre'].tolist()) 
Objtype_class_num,Objtype_class_list,Objtype_class_dict,Objtype_class_weight=generate_classdict(y)
# y=np.array(dataset['Creation Date'].tolist()) 
# CreationDate_class_num,CreationDate_class_list,CreationDate_class_dict,CreationDate_class_weight=generate_classdict(y)
y=np.array(dataset['artist'].tolist()) 
Artist_class_num,Artist_class_list,Artist_class_dict,Artist_class_weight=generate_classdict(y)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
print(sss.get_n_splits(X, y))
train_frame=pd.DataFrame()
test_frame=pd.DataFrame()
for train_index, test_index in sss.split(X, y):
  train_frame=dataset.loc[train_index]
  test_frame=dataset.loc[test_index]

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
path='/content/images/'
train_input_shape = (224,224)
batch_size=64
imgs_size=(64,224,224,3)
Artist_size=(batch_size,Artist_class_num)
Style_size=(batch_size,Style_class_num)
Objtype_size=(batch_size,Objtype_class_num)
# CreationDate_size=(batch_size,CreationDate_class_num)
def multi_task_Gen():
  iter=train_frame.iterrows()
  x_array=np.zeros(imgs_size)
  y1_array=[]
  y2_array=[]
  y3_array=[]
  y4_array=[]
  count=0
  while True:
    if count>=batch_size:
      
      x_array=np.asarray(x_array)
      y1_array=np.asarray(y1_array)
      y2_array=np.asarray(y2_array)
      y3_array=np.asarray(y3_array)
      # y4_array=np.asarray(y4_array)
      # print(x_array.shape)
      # print(y1_array.shape)
      # print(y2_array.shape)
      # print(y3_array.shape)
      
      # print(np.array([y1_array,y2_array,y3_array]).shape)
      yield x_array,{'Artist_output':y1_array,'Style_output':y2_array,'Objtype_output':y3_array}#,'CreationDate_output':y4_array
      count=0
      x_array=np.zeros(imgs_size)
      y1_array=[]
      y2_array=[]
      y3_array=[]
      # y4_array=[]
    dataframe = next(iter)
    # print()
    #print(to_categorical(class_dict[dataframe[1]['Artist']],num_classes=n_class))
    x_array[count]=(img_to_array(load_img(path+dataframe[1]['filename'],target_size=train_input_shape))*1./255)
    #print(count)
    y1_array.append(to_categorical(Artist_class_dict[dataframe[1]['artist']],num_classes=Artist_class_num))
    y2_array.append(to_categorical(Style_class_dict[dataframe[1]['style']],num_classes=Style_class_num))
    y3_array.append(to_categorical(Objtype_class_dict[dataframe[1]['genre']],num_classes=Objtype_class_num))
    # y4_array.append(to_categorical(CreationDate_class_dict[dataframe[1]['Creation Date']],num_classes=CreationDate_class_num))
    #print(dataframe[1]['Style'],'//',dataframe[1]['Object Type'])
    count+=1
def multi_task_Gen_valid():
  iter=test_frame.iterrows()
  x_array=np.zeros(imgs_size)
  y1_array=[]
  y2_array=[]
  y3_array=[]
  # y4_array=[]
  count=0
  while True:
    if count>=batch_size:
      
      x_array=np.asarray(x_array)
      y1_array=np.asarray(y1_array)
      y2_array=np.asarray(y2_array)
      y3_array=np.asarray(y3_array)
      # y4_array=np.asarray(y4_array)
      # print(x_array.shape)
      #print(np.array([y1_array,y2_array,y3_array]).shape)
      yield x_array,{'Artist_output':y1_array,'Style_output':y2_array,'Objtype_output':y3_array}#,'CreationDate_output':y4_array
      count=0
      x_array=np.zeros(imgs_size)
      y1_array=[]
      y2_array=[]
      y3_array=[]
      # y4_array=[]
    dataframe = next(iter)
    #print(to_categorical(class_dict[dataframe[1]['Artist']],num_classes=n_class))
    x_array[count]=(img_to_array(load_img(path+dataframe[1]['filename'],target_size=train_input_shape))*1./255)
    #print(count)
    y1_array.append(to_categorical(Artist_class_dict[dataframe[1]['artist']],num_classes=Artist_class_num))
    y2_array.append(to_categorical(Style_class_dict[dataframe[1]['style']],num_classes=Style_class_num))
    y3_array.append(to_categorical(Objtype_class_dict[dataframe[1]['genre']],num_classes=Objtype_class_num))
    # y4_array.append(to_categorical(CreationDate_class_dict[dataframe[1]['Creation Date']],num_classes=CreationDate_class_num))
    #print(dataframe[1]['Style'],'//',dataframe[1]['Object Type'])
    count+=1
train_generator = tf.data.Dataset.from_generator(
     multi_task_Gen,
     (tf.float64, {'Artist_output':tf.float32,'Style_output':tf.float32,'Objtype_output':tf.float32}),#,'CreationDate_output':tf.float32
     (imgs_size, {'Artist_output':Artist_size,'Style_output':Style_size,'Objtype_output':Objtype_size,}))#'CreationDate_output':CreationDate_size
valid_generator = tf.data.Dataset.from_generator(
     multi_task_Gen_valid,
     (tf.float64, {'Artist_output':tf.float32,'Style_output':tf.float32,'Objtype_output':tf.float32}),#,'CreationDate_output':tf.float32
     (imgs_size, {'Artist_output':Artist_size,'Style_output':Style_size,'Objtype_output':Objtype_size}))#,'CreationDate_output':CreationDate_size
#tf.TensorShape
#Load pre-train model
from tensorflow.keras.applications import *
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
import tensorflow as tf
import os
train_input_shape=(224,224,3)
based_model = ResNet50(weights='imagenet', include_top=False, input_shape=train_input_shape)
for layer in based_model.layers:
    layer.trainable = True
# Add layers at the end
X = based_model.output
X = Flatten()(X)

X = Dense(512, kernel_initializer='he_uniform')(X)
#X = Dropout(0.5)(X)
X = BatchNormalization()(X)
X = Activation('relu')(X)

X = Dense(125, kernel_initializer='he_uniform')(X)
#X = Dropout(0.5)(X)
X = BatchNormalization()(X)
X = Activation('relu')(X)

output1 = Dense(Artist_class_num, name='Artist_output',activation='softmax')(X)
output2 = Dense(Style_class_num, activation='softmax',name='Style_output')(X)
output3 = Dense(Objtype_class_num, activation='softmax',name='Objtype_output')(X)
# output4 = Dense(CreationDate_class_num, activation='sigmoid',name='CreationDate_output')(X)
model = Model(inputs=based_model.input, outputs=[output1,output2,output3])#,output4
optimizer = Adam(lr=1e-4)
model.compile(loss={'Artist_output': 'categorical_crossentropy', 'Style_output': 'categorical_crossentropy', 'Objtype_output': 'categorical_crossentropy'},#,'CreationDate_output':'mean_squared_error'
              optimizer=optimizer,
              loss_weights={'Artist_output':1,'Style_output':0.3,'Objtype_output':0.3},#,'CreationDate_output':0.3
              metrics={'Artist_output':'accuracy','Style_output':'accuracy','Objtype_output':'accuracy'})#,'CreationDate_output':'accuracy'
n_epoch=10
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
history1=model.fit(train_generator,
                  validation_data = valid_generator,
                  epochs=n_epoch,
                  shuffle=True,
                  verbose = 1,
                  use_multiprocessing=True,
                  callbacks=[tensorboard_callback],
                  workers=16,)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
def plot_metrics(history):
  metrics =  ['loss', 'accuracy']
  outputs=['Artist_output','Style_output','Objtype_output']#,'CreationDate_output'
  for n, output in enumerate(outputs):
    metric=metrics[0]
    name = output.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,3,n+1)
    plt.plot(history.epoch,  history.history[output+'_'+metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+output+'_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.suptitle(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    metric=metrics[1]
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,3,n+4)
    plt.plot(history.epoch,  history.history[output+'_'+metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+output+'_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.suptitle(name)

    plt.legend()
    plt.savefig('training_plot.png')
plot_metrics(history1)
save_dir = 'saved_models'
model_name = 'resnet50_art.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, 
                           mode='auto', restore_best_weights=True)
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)
for layer in model.layers[:50]:
    layer.trainable = False                               
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, 
                              verbose=1, mode='auto')
optimizer = Adam(lr=1e-5)
model.compile(loss={'Artist_output': 'categorical_crossentropy', 'Style_output': 'categorical_crossentropy', 'Objtype_output': 'categorical_crossentropy'},#,'CreationDate_output':'mean_squared_error'
              optimizer=optimizer,
              loss_weights={'Artist_output':1,'Style_output':0.3,'Objtype_output':0.3},#,'CreationDate_output':0.3
              metrics={'Artist_output':'accuracy','Style_output':'accuracy','Objtype_output':'accuracy'})#,'CreationDate_output':'accuracy'
#n_epoch=10
history1=model.fit(train_generator,
                  validation_data = valid_generator,
                  epochs=n_epoch,
                  shuffle=True,
                  verbose = 1,
                  use_multiprocessing=True,
                  callbacks=[reduce_lr,early_stop,checkpoint,tensorboard_callback],
                  workers=16,)
