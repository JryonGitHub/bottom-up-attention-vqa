from __future__ import print_function
import os
import json
import cPickle
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable




def test(model, test_loader, output, dataroot='data'):
    utils.create_dir(output)
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    model.load_state_dict(torch.load('/media/jry/MyBook/VQA/bottom-up-attention-vqa/saved_models/exp0/model.pth'))
    ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
    label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
    ans2label = cPickle.load(open(ans2label_path, 'rb'))
    label2ans = cPickle.load(open(label2ans_path, 'rb'))


    result = []
    for i, (qes_ids, img_ids, v, b, q) in enumerate(test_loader):
        v = Variable(v).cuda()
        b = Variable(b).cuda()
        q = Variable(q).cuda()

        pred = model(v, b, q, None)
        logits = torch.max(pred, 1)[1].data # argmax
        for qes_id, logit in zip(qes_ids, logits):
            result.append({
                'answer': label2ans[logit],
                'question_id': qes_id
            })

    outfile = os.path.join(output, 'result.json')
    with open(outfile, 'w') as f:
        json.dump(result, f)
    print('Generated %d outputs, saving to %s' % (len(result), outfile))
