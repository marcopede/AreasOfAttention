#save images and object proposals in memory for training

try:
    import os,subprocess
    gpu_id = subprocess.check_output('gpu_getIDs.sh', shell=True)
    os.environ["THEANO_FLAGS"]='device=gpu%s'%gpu_id
    print(os.environ["THEANO_FLAGS"])
except:
  pass


#load images as a dictionary
import glob

import numpy as np
from psutil import virtual_memory
#from cStringIO 
import StringIO
from scipy.io import loadmat
import json
#from PIL import Image
import string

def tokenize(sentence):
    import nltk
    return nltk.word_tokenize(str(sentence).lower().translate(None,string.punctuation))    

def convertChar(c):
    c1=c.lower().translate(None,string.punctuation)
    if c1==' ':
        n = 0
    elif c1>='a' and c1<='z':
        n = ord(c1)-ord('a')+1
    elif c1>='0' and c1<='9':
        n = ord(c1)-ord('0')+ord('z')-ord('a')+1
    else:
        n= ord('z')-ord('a')+11
    return n

def deconvertChar(n):
    if n==0:
        c = ' '
    elif n>=1 and n<=ord('z')-ord('a')+1:
        c = chr(ord('a')-1+n)
    elif n>=ord('z')-ord('a')+2 and n<=ord('z')-ord('a')+12:
        c = chr(n+ord('0')+ord('a')-ord('z')+2)
    else:
        c= 'X'
    return c

def convertSent(sentence,length):
    output = np.zeros(length,dtype=np.int8) #but using only 6 bits used    
    sent = str(sentence).lower().translate(None,string.punctuation)
    sentint = [convertChar(x) for x in sent[:length]]
    output[:len(sentint)]=sentint
    return output

def convertSentEnd(sentence):
    sent = str(sentence).lower().translate(None,string.punctuation)
    sentint = [convertChar(x) for x in sent][:sent.find('   ')]
    return sentint

def deconvertSent(numbers):
    sent = [deconvertChar(x) for x in numbers]
    return ''.join(sent)

def convert(file_im):
    from PIL import Image
    cim = Image.open(file_im)
    cbuffer = StringIO.StringIO()
    cim.save(cbuffer,'JPEG')
    return cbuffer

def deconvert(cbuffer):
    from PIL import Image
    cim = Image.open(cbuffer)
    npim = np.asarray(cim)
    return npim

def deconvertMat(cbuffer):
    from PIL import Image
    cbuffer = StringIO.StringIO(str(bytearray(cbuffer)))
    cim = Image.open(cbuffer)
    npim = np.asarray(cim)
    return npim

def convertNonUnicode(input):
    if isinstance(input, dict):
        return {convert(key): convert(value) for key, value in input.iteritems()}
    elif isinstance(input, list):
        return [convert(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input

def convertMat(voc):
    newvoc={}
    for k,v in voc.items():
        newvoc[str(k)]=voc[k]
        voc[k]['image']=np.array(bytearray(voc[k]['image'].getvalue()))
        #to get it back use stringIO(str(bytearray(x[0][0][0])))
    return newvoc



def loop_images(img_path='coco/train2014/',
                boxes_path='/home/lear/mpederso/links/scratch1/data/coco/precomputed-coco/edge_boxes_70/mat/COCO_train2014/',
                ann_path='coco/annotations/'):
    imgs = glob.glob(img_path+'*.jpg')
    imgs.sort()
    if img_path.find('train')>=0:
        split='train'
    else:  
        split='val'
    annotations=json.load(open(ann_path+'captions_%s2014.json'%split))['annotations']
    data = {}
    for idann,ann in enumerate(annotations):
        if idann%1000==0:
            print idann*100/len(annotations),"%"
        if data.has_key(ann['image_id']):
            data[ann['image_id']]['caption'].append(ann['caption'])
        else:
            data[ann['image_id']]={'caption':[ann['caption']]}
    for idim,im in enumerate(imgs):
        imnum=int(im.split('/')[-1].split('_')[-1].split('.')[0])
        if idim%1000==0:
            print idim*100/len(imgs),"%",im
            print 'Memory',virtual_memory().percent
        if split=='train':
            boxes=loadmat(boxes_path+'COCO_%s2014_%07d/COCO_%s2014_%012d'%(split,imnum/100000,split,imnum))
        else:
            boxes=loadmat(boxes_path+'COCO_%s2014_%09d/COCO_%s2014_%012d'%(split,imnum/1000,split,imnum))
        data[imnum]['boxes']=boxes['boxes']
        try:
            data[imnum]['image']=convert(im)
        except IOError:
            print "Loading problems for image ",im
            print "Skipping it!"
        #print 'Memory',virtual_memory().percent
    #check that every image has the three fields
    ndel=0
    for key,dd in data.items():
        if not dd.has_key('caption'): 
            print "Image missing caption, deleting",key
            del data[key]
            ndel+=1
        elif not dd.has_key('image'):
            print "Image missing image, deleting",key
            del data[key]
            ndel+=1
        elif not dd.has_key('boxes'):
            print "Image missing boxes, deleting",key
            del data[key]
            ndel+=1
    if ndel>0:
        print "Warning, deleted", ndel,'Images, because missing annotations!'
    return data
   
if __name__ == "__main__":
    import pylab

    #training data
    dbtrain=loop_images()
    from scipy.io import savemat
    #savemat(dbtrain,'coco/dbtrain.mat')
    #import cPickle as pickle
    #convert int into strings
    
    print'Saving coco/dbtrain.mat'
    #savemat(newdbtrain,'coco/dbtrain.mat')

    #pickle.dump(dbtrain,open('coco/train.pkl','wb'))
    del dbtrain

    #validation data
    dbval=loop_images(img_path='coco/val2014/',boxes_path='/home/lear/mpederso/links/scratch1/data/coco/precomputed-coco/edge_boxes_70/mat/COCO_val2014_0/')
    newdbval={}
    for k,v in dbtrain.items():
        newdbval[str(k)]=dbval[k]
    print'Saving coco/val.pkl'
    #savemat(dbtrain,'coco/dbval.mat')
    #pickle.dump(dbval,open('coco/val.pkl','wb'))


