import pickle
import argparse
import glob
import os
import numpy as np
import pdb

parser = argparse.ArgumentParser(
        description='cvppa convert to submission file.')

parser.add_argument('--pred-dir', help='prediction pkl files dir')
parser.add_argument('--pred-fnames', nargs='+', help='prediction pkl filenames')
parser.add_argument('--out-path', help='output txt path')
args = parser.parse_args()




#pred_paths = glob.glob(os.path.join(args.pred_dir,'pred_*.pkl'))
pred_paths = []
for pred_fname in args.pred_fnames:
    pred_paths.append(os.path.join(args.pred_dir,pred_fname))


results = []
for pred_path in pred_paths:
    result = []
    inames = []
    with open(pred_path,'rb') as f:
        data = pickle.load(f)
        for i in range(len(data)):
            pred_score = data[i]['pred_score'].numpy()
            img_path = data[i]['img_path']
            img_name = img_path.split('/')[-1]
            result.append(pred_score)
            inames.append(img_name)
        results.append(result)

#pdb.set_trace()

preds = []
for i in range(len(inames)):
    score = np.zeros(7)
    for j in range(len(results)):
        score += results[j][i]
    score = score / j
    pred = np.argmax(score)
    preds.append(pred)
    
with open(args.out_path,'w') as f:
    for i in range(len(inames)):
        f.write(inames[i] + ' ' + str(preds[i]) + '\n')    
        
print('Finished.')
    

        