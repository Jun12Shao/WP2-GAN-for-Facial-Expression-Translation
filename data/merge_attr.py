import pickle
import os
import random
import numpy as np






# path="/mnt/jun/Datasets/meta-data/4dts_train_ids.pkl"
#
#
#
# with open(path,"rb") as f:
#     list=pickle.load(f,encoding='latin1')
#     f.close()
# bad_images=['/AGFW/AGFW_128/m_30_34_pic_0477','/AGFW/AGFW_128/m_30_34_pic_0478']
# for img in bad_images:
#     list.remove(img)
#
# with open(path,'wb') as f:
#     pickle.dump(list,f)
#     f.close()





# path="/mnt/jun/Datasets/meta-data/meta-4-datasets/EM_aus_pose_age_gender_attr(90%)_2.pkl"
#
#
#
# with open(path,"rb") as f:
#     dictionary=pickle.load(f,encoding='latin1')
#     f.close()
#
# for id in dictionary:
#     cond=np.concatenate((dictionary[id][:17],dictionary[id][19:]),axis=None)
#
#     dictionary[id]=cond
#
# with open("/mnt/jun/Datasets/meta-data/meta-4-datasets/EM_aus_pose_age_gender_attr(90%)_2.pkl",'wb') as f:
#     pickle.dump(dictionary,f)
#     f.close()







# root="/home/sh_jun/Datasets"
attr_dir="/mnt/jun/Datasets/meta-data/meta-4-datasets"
attr_files=["AGFW_aus_pose_age_gender_attr(90%).pkl","CACD_aus_pose_age_gender_attr(90%).pkl","CFEE_aus_pose_age_gender_attr(90%).pkl","EM_aus_pose_age_gender_attr(90%)_0.pkl","EM_aus_pose_age_gender_attr(90%)_1.pkl","EM_aus_pose_age_gender_attr(90%)_2.pkl"]

predix=["/AGFW_128/","/CACD_128/","/CFEE_128/","/EmotioNet_128_0/","/EmotioNet_128_1/","/EmotioNet_128_2/"]


#
# with open(attr_dir+'/aus_age_gender_label.pkl','rb') as f:
#     dict=pickle.load(f,encoding='latin1')
#     f.close()
#
#
# dict={}

for i in range(6):
    attr_path= os.path.join(attr_dir,attr_files[i])
    with open(attr_path,"rb") as f:
        dictionary=pickle.load(f,encoding='latin1')
        f.close()
    if i==0:
        bad_images=['m_30_34_pic_0477','m_30_34_pic_0478']
        for img in bad_images:
            del dictionary[img]


    for id in dictionary:
        cond=dictionary[id]
        new_id=predix[i]+id
        # dictionary[new_id]=dictionary.pop(id)
        dict[new_id]=dictionary[id]
    # dict.update(dictionary)
    print (len(dict))

with open(attr_dir+'/aus_age_gender_label.pkl','wb') as f:
    pickle.dump(dict,f)
    f.close()

name_list=list(dict.keys())
random.shuffle(name_list)

length=len(name_list)
k=int(0.9*length)
namelist_train=name_list[:k]
namelist_test=name_list[k:]


with open(attr_dir+'/4dts_train_ids.pkl','wb') as f:
    pickle.dump(namelist_train,f)
    f.close()

with open(attr_dir + '/4dts_test_ids.pkl', 'wb') as f:
    pickle.dump(namelist_test, f)
    f.close()

