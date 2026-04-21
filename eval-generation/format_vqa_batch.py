import os
import json
import sys
from utils.constants import QA_DIR
os.makedirs(QA_DIR, exist_ok=True)

data_template = {"question_id": "",
            "image": "",
            "text": "",
            "category": ""}

def pathmmu_format_test(datadir, outf):
    data = json.load(open(f'{datadir}/pathmmu.json', 'r'))
    imgdir = f'{datadir}/images'
    num_total = 0
    for key in data.keys():
        for sp in ['test', 'test_tiny']:
            num_total += len(data[key][sp])
            if key == 'Atlas':
                num_img = 0
                for d in data[key][sp]:
                    if os.path.exists(f'{imgdir}/{d["img"]}'):
                        num_img += 1
                print(key, sp, len(data[key][sp]), num_img)
    print('total', num_total)
    # Two options: 8670 Test or 9300 Test+Val  
    keys = ['PubMed', 'SocialPath', 'EduContent', 'PathCLS']
    split = ['val', 'test', 'test_tiny']
    data_whole = []
    for sp in split:
        for key in keys:
            datalist = data[key][sp]
            for idx, d in enumerate(datalist):
                d['img'] = f'{imgdir}/{d["img"]}'
                assert os.path.exists(d['img']), d['img']
                assert str(d['No']) == str(idx), (d['No'], idx)
                d['No'] = '-'.join([key, sp, str(idx)])
            print(key, sp, len(datalist))
            data_whole += datalist

    print(len(data_whole))
    with open(outf, 'w') as outfile:
        for _data in data_whole:
            data = data_template.copy()
            data['question_id'] = _data['No']
            data['category'] = 'mc'
            data['image'] = _data['img']
            data['text'] = _data['question'] +'. '.join(_data['options'])+'.'
            data['answer'] = _data['answer']
            jout = json.dumps(data) + '\n'
            outfile.write(jout)


if __name__ == '__main__':
    datadir = sys.argv[1]
    dataset = sys.argv[2]
    format_file = f'{QA_DIR}/ques_{dataset}.jsonl'
    
    
    # Save formatted file
    if dataset == 'pathmmu':
        pathmmu_format_test(datadir, format_file)