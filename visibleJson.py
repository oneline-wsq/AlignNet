import json
import os
import matplotlib.pyplot as plt

color={0:'red',1:'green',2:'blue'}
def visibleJson(json_name):
    with open(json_name,'r',encoding='utf8')as fp:
        json_data = json.load(fp)
    results=json_data['results']
    for key, values in results.items():
        lineNums=len(values)
        for line in values:
            pts = line['pts']
            pts_num = line['pts_num']
            line_type = line['type']
            confidence_level  = line['confidence_level']
            
            # 画图
            plt.plot(pts[0],pts[1],color=color[line_type],linewidth=0.5)
        plt.savefig(f'imgs/vetorize/{key}.png')

if __name__ =='__main__':
    visibleJson('output.json')
    print('end')