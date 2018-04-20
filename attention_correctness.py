# coding: utf-8
#load data
from scipy.io import loadmat as lm
import cPickle as pickle
import nltk
import numpy as np
import pylab
from scipy.ndimage import zoom
import matplotlib
from skimage.draw import polygon
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--draw', action='store_true', help='Draw the attention correctness for each image')
parser.add_argument('--mode',dest='mode',default='grid', help='Region proposal mode: grid,prop,transf')
parser.add_argument('--reduced', action='store_true', help='Use a reduced version with only 50 proposals instead of 200')
parser.add_argument('--ongt', action='store_true', help='Evaluate on ground truth sentences')
parser.add_argument('--att_path', type=str, default=None, help='Path to the saved attention maps to evaluate')
parser.add_argument('--rescale_prob', action='store_true', help='Clip and rescale the prop density. Improves results but makes process different from prior art so off by default.')
args = parser.parse_args()
params = vars(args) # convert to ordinary dict

draw=params['draw']
mode=params['mode']#'prop'
reduced=params['reduced']
ongt = params['ongt']
att_path = params['att_path']
#mode='transf'
if reduced:
    dset='reduced'
else:
    dset='test'

if ongt:
    dset = 'gt'

import glob
#mode='grid'
if att_path is None:
    print("You did not specify an att_path, Is that normal? Marco has a default strategy, but if your name is Thomas, that's very probably a mistake.")
    # Just remove this is you are called Marco and it's nagging at you 
    import pdb; pdb.set_trace()
    if mode=='grid':
        #imgs = glob.glob('flickr30k/test_grid/*.pkl')
        imgs = glob.glob('flickr30k/'+dset+'_grid/*.pkl')
    elif mode=='transf':
        #imgs = glob.glob('flickr30k/test_transf/*.pkl')
        imgs = glob.glob('flickr30k/'+dset+'_transf/*.pkl')
    elif mode=='prop':
        #test = reduced up to 460
        #normal = test
        #reduced = reduced recomputing
        if ongt:
            imgs = glob.glob('flickr30k/gt_prop/*.pkl')
        else:
            imgs = glob.glob('flickr30k/normal_prop/*.pkl')
else:
    imgs = glob.glob(att_path+'*.pkl')

#build a vocabulary with image name as key
if os.path.exists('flickr30k/annotations.dict'):
    print('Loading Precomputed Annotations')
    images = pickle.load(open('flickr30k/annotations.dict'))
else:
    ann = lm('flickr30k/Flickr30kEntities/annotations.mat')['ann'][0]
    sent = lm('flickr30k/Flickr30kEntities/sentences.mat')['ann'][0]
    iomages = {}
    for idl,l in enumerate(ann):
        images[l[0]['image'][0][0]]={'sentence':[x['sentence'][0][0] for x in sent[idl]],'phrases':sent[idl][0]['phrases'],'phraseFirstWordIdx':sent[idl]['phraseFirstWordIdx'],'phraseID':sent[idl][0]['phraseID'],'id':l[0]['id'][0][0],'boxes': l[0]['labels'][0]['boxes'],'idToLabel':l[0]['idToLabel'][0][0],'id2lab':{x[0][0]:l[0]['idToLabel'][0][idx][0] for idx,x in enumerate(l[0]['id'][0])}}
    pickle.dump(images,open('flickr30k/annotations.dict','wb'))


def inter(ba,bb):
    bmaxy=min(ba[3],bb[3])
    bmaxx=min(ba[2],bb[2])
    bminy=max(ba[1],bb[1])
    bminx=max(ba[0],bb[0])
    if bmaxy<bminy or bmaxx<bminx:
        return [0,0,0,0]
    else:
        return [bminx,bminy,bmaxx,bmaxy]

def rescale(box,vim,new_size):
    new_b=[0,0,0,0]
    new_b[0]=int(np.round(box[0]/float(vim[1])*new_size[1]))
    new_b[2]=int(np.round(box[2]/float(vim[1])*new_size[1]))
    new_b[1]=int(np.round(box[1]/float(vim[0])*new_size[0]))
    new_b[3]=int(np.round(box[3]/float(vim[0])*new_size[0]))
    return new_b

def rescale2(box,vim,new_size):
    h = vim[0]
    w = vim[1]
    factor = min(w,h)
    new_b=[0,0,0,0]
    if w>h:
        new_b[0]=((box[0]-(w-h)/2.0)/float(factor)*new_size[1])
        new_b[2]=((box[2]-(w-h)/2.0)/float(factor)*new_size[1])
        new_b[1]=(box[1]/float(factor)*new_size[0])
        new_b[3]=(box[3]/float(factor)*new_size[0])
    else:
        new_b[0]=(box[0]/float(factor)*new_size[1])
        new_b[2]=(box[2]/float(factor)*new_size[1])
        new_b[1]=((box[1]-(h-w)/2.0)/float(factor)*new_size[0])
        new_b[3]=((box[3]-(h-w)/2.0)/float(factor)*new_size[0])
    new_b[0] =int(np.round(new_b[0]).clip(0,new_size[1]))
    new_b[2] =int(np.round(new_b[2]).clip(0,new_size[1]))
    new_b[1] =int(np.round(new_b[1]).clip(0,new_size[0]))
    new_b[3] =int(np.round(new_b[3]).clip(0,new_size[0]))
    return new_b

def area(sumpratt,box):
    #return sumpratt[box[2],box[3]]-sumpratt[box[2],box[1]]-sumpratt[box[0],box[3]]+2*sumpratt[box [0],box[1]]
    return sumpratt[box[3],box[2]]-sumpratt[box[1],box[2]]-sumpratt[box[3],box[0]]+2*sumpratt[box [1],box[0]]

def area2(pratt,box):
    return pratt[box[1]:box[3],box[0]:box[2]].sum()

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def poly(box,prob):
    pol=Polygon(box,True,alpha=prob)
    return pol

imgs.sort()
#imgs = imgs[:500]

#loop over images
lpr=[]
for idim,im in enumerate(imgs):
    imid = im.split('/')[-1].split('.')[0]
    gt = images[imid]
    fs = open(im)
    gen = pickle.load(fs)
    #gensent = ' '.join(gen['word'])
    if dset == 'gt':
        pos = nltk.pos_tag(gt['sentence'][0].split(' ')[:21])
    else:
        pos = nltk.pos_tag(gen['word'])
    oldphrid = -1
    for (idl,(l,p)) in enumerate(pos):
        if p=='NN' or p=='NNS':
            #find word in annotations
            Nobox=False
            for pss,ss in enumerate(gt['phrases']): #for each sententence
                for pphr,phr in enumerate(ss[0]):
                    if l.lower() in phr:
                        #plot figure
                        if draw:
                            pylab.figure(1)
                            pylab.clf()
                        vim=pylab.imread('flickr30k/flickr30k/imgs/'+imid+'.jpg')
                        if draw:
                            pylab.imshow(vim)
                        phrid = gt['phraseID'][pss][0][pphr][0]
                        #plot bboxes
                        #bb=gt['idToLabel'][idphr][0]
                        if gt['id2lab'].has_key(phrid):
                            bb=gt['id2lab'][phrid]
                            for b in bb:
                                if len(gt['boxes'][b[0]-1][0])>0:
                                    box = gt['boxes'][b[0]-1][0][0]
                                    pylab.plot([box[0],box[2],box[2],box[0],box[0]],[box[1],box[1],box[3],box[3],box[1]],lw=3)
                                else:
                                    print('No Visible BBox!')
                                    Nobox = True
                            if draw:
                                pylab.draw()
                                pylab.show()
                            #plot att
                            att = pickle.load(open(im))
                            rtype='occupancy'#'point'
                            if rtype=='occupancy':
                                if draw:
                                    pylab.figure(2)
                                    pylab.clf()
                                #dx = vim.shape[1]/224.0
                                #dy = vim.shape[0]/224.0
                                new_size = (224,224)
                                if mode=='prop':
                                    new_size = (450,450)
                                pratt = np.zeros(new_size)
                                for idb,box in enumerate(att['boxes']):
                                    rr,cc=polygon(box[1],box[0])
                                    rr = rr.clip(0,new_size[1]-1)
                                    cc = cc.clip(0,new_size[0]-1)
<<<<<<< HEAD

                                    # If true, improves results but makes results less comparable with prior art.
                                    should_rescale_prob = params['rescale_prob'] and mode in ['prop', 'transf', 'grid']
                                    if should_rescale_prob:
=======
                                    if 0:#give slightly better results by making the distribution more peaky
>>>>>>> 015428ec4bca2df04d0b82960ca666ccfe8c4020
                                        pratt[rr,cc]+=att['prob'][idl][idb]**2/float(len(rr))
                                    else:
                                        pratt[rr,cc]+=att['prob'][idl][idb]#/float(len(rr))
                                pratt /=pratt.sum()
                                if draw:
                                    pylab.imshow(pratt)

                                pr = 0
                                scl = min(vim.shape[0],vim.shape[1])/300.0
                                if max(vim.shape[0],vim.shape[1])/scl>450:
                                    scl = max(vim.shape[0],vim.shape[1])/450.0
                                print "Original image",vim.shape
                                print "Rescaled image",np.array(vim.shape)/float(scl)
                                #raw_input()

                                for b in bb:
                                    if len(gt['boxes'][b[0]-1][0])>0:
                                        obox = gt['boxes'][b[0]-1][0][0]
                                        if mode=='prop':
                                            #scl=max(vim.shape[0],vim.shape[1])
                                            #box = rescale(obox,[scl,scl],new_size)
                                            print "scale",scl
                                            box = rescale(obox,(scl,scl),(1,1))
                                        else:
                                            box = rescale2(obox,vim.shape,new_size)
                                        if draw:
                                            pylab.plot([box[0],box[2],box[2],box[0],box[0]],[box[1],box[1],box[3],box[3],box[1]],'r',lw=3)
                                        pr += area2(pratt,box)#sumpratt[box[2],box[3]]
                                        #pr += area(sumpratt,box)
                                        print "Attention Correctness",pr
                                        for other_b in bb:
                                            if len(gt['boxes'][other_b[0]-1][0])>0:
                                                if b!=other_b:
                                                    other_box = gt['boxes'][other_b[0]-1][0][0]
                                                    bint=inter(obox,other_box)
                                                    if mode=='prop':
                                                        #scl=max(vim.shape[0],vim.shape[1])
                                                        #other_box = rescale(bint,[scl,scl],new_size)
                                                        #scl = min(vim.shape[0],vim.shape[1])/300.0
                                                        #if max(vim.shape[0],vim.shape[1])*scl>450:
                                                        #    scl = max(vim.shape[0],vim.shape[1])/450.0
                                                        other_box = rescale(bint,(scl,scl),(1,1))
                                                    else:
                                                        other_box = rescale2(bint,vim.shape,new_size)
                                                    pr -= 0.5*area2(pratt,other_box)
                                                    #pr -= 0.5*area(sumpratt,other_box)
                                                    print("Overlap",b[0]-1,other_b[0]-1,area2(pratt,other_box))
                                        print "Attention Correctness",pr
                                        #lpr.append(pr)
                            if rtype=='boxes':
                                dx = vim.shape[1]/224.0
                                dy = vim.shape[0]/224.0
                                for idb,box in enumerate(att['boxes']):
                                    if draw:
                                        pylab.plot(box[0]*dx,box[1]*dy,'r',lw=att['prob'][idl][idb]*30)
                            elif rtype=='point':
                                if draw:
                                    pylab.figure(2)
                                    pylab.clf()
                                pratt= att['prob'][idl].reshape(14,14)
                                zm = 10
                                new_size = (pratt.shape[0]*zm,pratt.shape[1]*zm)
                                sumpratt = np.zeros((new_size[0]+1,new_size[1]+1))
                                pratt = zoom(pratt, (zm,zm), order=1)
                                pratt /= pratt.sum()
                                if draw:
                                    pylab.imshow(pratt)
                                sumpratt[1:,1:]=pratt.cumsum(0).cumsum(1)
                                pr = 0
                                for b in bb:
                                    if len(gt['boxes'][b[0]-1][0])>0:
                                        obox = gt['boxes'][b[0]-1][0][0]
                                        box = rescale(obox,vim.shape,new_size)
                                        pylab.plot([box[0],box[2],box[2],box[0],box[0]],[box[1],box[1],box[3],box[3],box[1]],'r',lw=3)
                                        pr += area2(pratt,box)#sumpratt[box[2],box[3]]
                                        #pr += area(sumpratt,box)#sumpratt[box[2],box[3]]-sumpratt[box[2],box[1]]-sumpratt[box[0],box[3]]+2*sumpratt[box[0],box[1]]
                                        print "Attention Correctness",pr
                                        for other_b in bb:
                                            if len(gt['boxes'][other_b[0]-1][0])>0:
                                                if b!=other_b:
                                                    other_box = gt['boxes'][other_b[0]-1][0][0]
                                                    bint=inter(obox,other_box)
                                                    other_box = rescale(bint,vim.shape,new_size)
                                                    pr -= 0.5*area2(pratt,other_box)
                                                    #pr -= 0.5*area(sumpratt,other_box)#sumpratt[box[2],box[3]]-sumpratt[box[2],box[1]]-sumpratt[box[0],box[3]]+2*sumpratt[box[0],box[1]]
                                                    print("Overlap",b[0]-1,other_b[0]-1,area(sumpratt,other_box))
                                        print "Attention Correctness",pr
                            #evaluate overlap
                            print idim,l,phr
                            if draw:
                                pylab.draw()
                                pylab.show()
                            if not Nobox:
                                print "phrase ID",phrid
                                if phrid==oldphrid:
                                    pr = max(lpr[-1],pr)
                                    lpr[-1] = pr
                                    #raw_input()
                                else:
                                    lpr.append(pr)
                                    oldphrid=phrid
                                #lpr.append(pr)
                            print "Total Attention Correctness",np.mean(lpr)
                            if draw:
                                raw_input()


