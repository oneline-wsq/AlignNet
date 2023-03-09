import os
import json
import matplotlib.pyplot as plt
import numpy as np

colors_plt = ['#FF0000', '#00FF00', '#0000FF']
# 红，绿，蓝
#     'divider': 0,
#     'ped_crossing': 1,
#     'boundary': 2,
#     'others': -1

def tryLineString(json_name):
    # 读入json文件
    with open(json_name,'r',encoding='utf8') as fp:
        json_data=json.load(fp)
    
    token=json_name.split('/')[-1].split('.')[0]

    fig = plt.figure(figsize=(10, 5))
    ax=fig.add_subplot(1,1,1)

    line_num=0
    for k in json_data:
        line=k
        line_pts=line['pts']
        line_pts_num=line['pts_num']
        line_type=line['type']

        x=np.array([p[0] for p in line_pts])
        y=np.array([p[1] for p in line_pts])
        # if line_type==2:
        ax.plot(x, y, color=colors_plt[line_type], linewidth=1, alpha=0.5, markersize=1)
        line_num+=1 

    img_path=f'imgs/gt_differentType'
    if not os.path.exists(img_path):
        os.makedirs(img_path)  # 创建目录
    img_name=os.path.join(img_path,f'{token}.png')
    fig.savefig(img_name)
    plt.clf()
    plt.close()
    print(line_num)

if __name__=='__main__':
    json_name='dataset/gt_multiframes_box/1644541391216617.json'
    tryLineString(json_name)
    print('end')

