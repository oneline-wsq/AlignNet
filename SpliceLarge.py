import shutil
import os

def moveJson(sourcedir,newdir):
    jsons=sorted(os.listdir(sourcedir))
    for i in range(0,len(jsons),50):
        old_path=os.path.join(sourcedir,jsons[i])
        new_path=os.path.join(newdir,jsons[i])
        shutil.copy(old_path,new_path)

if __name__ == '__main__':
    gt_sourcedir='dataset/gt_multiframes_box'
    gt_newdir='dataset/vectorize_gt_box'

    pred_sourcedir='dataset/pred_multiframes_box'
    pred_newdir='dataset/vectorize_pred_box'
    
    moveJson(gt_sourcedir,gt_newdir)
    moveJson(pred_sourcedir,pred_newdir)

    print('end')