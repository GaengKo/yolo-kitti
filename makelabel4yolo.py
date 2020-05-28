import os

path = './images/training/image_02'
file_list = os.listdir(path)

f = open('./kitti.data','w')
for i in file_list:
    file_path = path+'/'+i
    file_dir = os.listdir(file_path)
    for j in file_dir:
        data = '.'+file_path+'/'+j+'\n'
        print(data)
        f.write(data)
f.close()
        


