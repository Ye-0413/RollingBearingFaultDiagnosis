import os, random, shutil
def moveFile(fileDir):
        pathDir = os.listdir(fileDir)    #取图片的原始路径
        filenumber=len(pathDir)
        rate=0.25    #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
        picknumber=int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
        sample = random.sample(pathDir, picknumber)  #随机选取picknumber数量的样本图片
        print (sample)
        for name in sample:
                shutil.move(fileDir+name, tarDir+name)
        return

if __name__ == '__main__':
	fileDir = "/Users/youngj/Local/机器学习缺陷检测/CWT_picture/load_3/Normai_3/"    #源图片文件夹路径
	tarDir = "/Users/youngj/Local/机器学习缺陷检测/CWT_picture/Cross_load/Cross_Normal/"   #移动到新的文件夹路径
	moveFile(fileDir)
