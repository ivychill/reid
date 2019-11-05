
import torch
import numpy as np
import os
import json

# sort the images
def sort_img(qf, gf):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]

    return index

def gen_submission(image_datasets, query_feature, gallery_feature, result_dir='./'):
    res = {}
    for k in range(len(query_feature)):
        index = sort_img(query_feature[k], gallery_feature)
        query_path, _ = image_datasets['query'].imgs[k]
        query_name = query_path.split('/')[-1]
        responses = []

        for i in range(200):
            img_path, _ = image_datasets['gallery'].imgs[index[i]]
            img_name = img_path.split('/')[-1]
            responses.append(img_name)

        res[query_name] = responses

    with open(os.path.join(result_dir, 'result.json'), 'w') as f:
        json.dump(res, f)