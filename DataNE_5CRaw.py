import numpy as np
import glob
import os
from skimage import io


pathDEM='Z:/CNN/Data/ClipedSample_4Areas/NE_WFBB/NEWFBB_DEM/' # DEM
pathA='Z:/CNN/Data/ClipedSample_4Areas/NE_WFBB/NEWFBB_Aer/' # Aerial photo

# read all images and labels
def read_raw_img(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)] # get F,T folder
    imgs=[]
    imgs_name=[]
    labels=[]
    for idx,folder in enumerate(cate): # idx-> 0:F; 1:T; folder-> F,T
   
        for im in glob.glob(folder+"/*.tif"):
            im = im.replace('\\','/')
            
            img_name=os.path.basename(im)
            img_name=os.path.splitext(img_name)[0] # get file name     

            img=io.imread(im) # read raw images
            
#            # Normalize the dataset-MaxMin
#            scaler = MinMaxScaler(feature_range=(0, 1))
#            img = scaler.fit_transform(img)
            
            
            imgs.append(img) # raw image data
            labels.append(idx) # label
            imgs_name.append(img_name) # image name
            
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32), np.asarray(imgs_name)


dataDEM, labelDEM, nameDEM = read_raw_img(pathDEM) # DEM->(num of samples, size, size)->3D
# print (dataDEM.shape,labelDEM.shape,nameDEM.shape)

dataA, labelA, nameA = read_raw_img(pathA) # Aerial photo->(num of samples, 100, 100, 4)->4D
# print (dataA.shape,labelA.shape,nameA.shape)


num=int(dataDEM.shape[0]) # total number of images



dataDEM4D=np.expand_dims(dataDEM,3) # add channel dimension,4D data
# dataA4D=np.expand_dims(dataA,3) # add channel dimension


data=np.zeros((num,100,100,5),dtype=np.float32) # 0:DEM, 1-4： RGB+NIR, 5 channels

print ('----------Read DEM iamges---------')

for i in range (num):
#    print (nameDEM[i],i)
    for j in range (100):
        for k in range (100):
            data[i,j,k,0]= dataDEM4D[i,j,k,0] # Read DEM in 1st D 

print ('----------Read Aerialphoto iamges---------')   
            
for i in range (num):
#    print (nameDEM[i],i)
    for p in range (len(nameA)):

    # match DEM with Aerial photos by using their names
        if (nameA[p]==(nameDEM[i])): 
#            print (nameA[p],p)
            for j in range (100):
                for k in range (100):
                    data[i,j,k,1]= dataA[p,j,k,0] # Read R band
                    data[i,j,k,2]= dataA[p,j,k,1] # Read G band
                    data[i,j,k,3]= dataA[p,j,k,2] # Read B band
                    data[i,j,k,4]= dataA[p,j,k,3] # Read NIR band      


label=labelDEM # label
name=nameDEM # name

print (nameA[46],nameDEM[1])

print (data[1,:,:,0])
print (dataDEM4D[1,:,:,0])

print (data[1,:,:,3])
print (dataA[1,:,:,2])


np.save("D:/CNNtest2022/Data4Area5C/WFBB_Rawdata5C.npy", data) # Raw data (num,100,100,5) images, 5 channels: 0-4 is DEM,RGB,NIR
np.save("D:/CNNtest2022/Data4Area5C/WFBB_label.npy", label)
np.save("D:/CNNtest2022/Data4Area5C/WFBB_name.npy",name)
