import os

#创建个文件夹
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("create path suss")
    else:
        print("path exist")
    return path
#创建个文件夹
path="../tb_info/"+"utils/1.txt"
create_folder(path)
