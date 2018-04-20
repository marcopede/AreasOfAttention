# coding: utf-8
# vim:fdm=marker

try:
    import os,subprocess
    gpu_id = subprocess.check_output('gpu_getIDs.sh', shell=True)
    os.environ["THEANO_FLAGS"]='device=gpu%s'%gpu_id
    print(os.environ["THEANO_FLAGS"])
except:
  pass
import glob
from scipy.io import savemat
import numpy as np
from psutil import virtual_memory
import StringIO
from scipy.io import loadmat, whosmat
import json
#from PIL import Image
import string
import os.path
import time 
import sys
import pickle as pkl
from PrepareData import tokenize, convertChar, deconvertChar, convertSent, convertSentEnd, deconvertSent, convert, deconvert, deconvertMat, convertNonUnicode, convertMat

def split_imgs(img_path, informations):
    # {{{
    """
    img_path: path to folder containing all *.jpb images.
    informations: list, for all images contains a fiels 'split' and a field 'filename'.
                  Elements 'filenames' are separated depending on 'split'.
    """

    imgs = glob.glob(img_path+'*.jpg')
    imgs.sort()
    train_filenames = [elem['filename'] for elem in informations if elem['split']=='train']
    test_filenames = [elem['filename'] for elem in informations if elem['split']=='test']
    val_filenames = [elem['filename'] for elem in informations if elem['split']=='val']
    assert(len(train_filenames + test_filenames + val_filenames) == len(informations)), \
        "Some images went missing: " + str(len(train_filenames + test_filenames + val_filenames)) + "," \
        + str(len(informations))

    train_imgs, test_imgs, val_imgs, = [], [], []
    completed = 0
    for idx, img in enumerate(imgs):
        if idx%5000 == 0:
            completed += 1
            print(str(completed*5000) + "/" +str(31000) + "... "),
            sys.stdout.flush()
        if img.split("/")[-1] in train_filenames:
            train_imgs.append(img)
        elif img.split("/")[-1] in test_filenames:
            test_imgs.append(img)
        elif img.split("/")[-1] in val_filenames:
            val_imgs.append(img)

    no_missing = (len(train_imgs) + len(test_imgs) + len(val_imgs) == len(informations))
    assert(no_missing), "Some images went missing: " + \
            str(len(train_imgs + test_imgs + val_imgs)) + \
            "," + str(len(imgs))
    print("Splits obtained!")
    # This really needs to be secure so we use a dict to force it.
    return {'train_imgs': train_imgs,
            'val_imgs': val_imgs,
            'test_imgs':test_imgs}
    # }}}

def loop_images(images,
                annotations,
                boxes_path=None):
    # {{{

    data = {}
    for idx, img in enumerate(images):
        print_interval = 500
        if idx % print_interval == 0:
            print(str(idx) +  "/" + str(len(images)) + "..."),
        img_id = img.split('/')[-1]
        ann = annotations[img_id]

        img_key = img_id.split('.')[0]
        assert(not data.has_key(img_key)), "For flickr preprocessing we don't expect to find many captions separately for the same image."
        data[img_key] = {'caption': ann}

        try:
            assert(not data[img_key].has_key('image')), "Image somehow already added."
            converted = convert(img)
            data[img_key].update({'image': convert(img)})
        except IOError:
            print ("Loading problems for image ",img)
            import pdb; pdb.set_trace()
            #print "Skipping it!"
    print("\n Finnished creating Data!")
    return data
    # }}}

def load_flickr(path='flick_data/', no_train=False, no_test=True, small=False):
    # {{{
    t = time.time()
    #variables = whosmat(os.path.join(path,'dbtrain.mat'))
    #compare_variables = whosmat(os.path.join('coco/','dbtrain.mat'))

#    dbtrain=loadmat(os.path.join(path,'dbtrain.mat'))
#    import pdb; pdb.set_trace()

    #dbtrain=loadmat(path+'dbtrain.mat')
    if not no_train:
        if small == True:
            dbtrain = pkl.load(open(os.path.join(path,'dbtrain_small.pkl')))
        else:
            dbtrain = pkl.load(open(os.path.join(path,'dbtrain.pkl')))
    else:
        dbtrain = None

    dbval=pkl.load(open(os.path.join(path,'dbval.pkl')))

    if not no_test:
        dbtest = pkl.load(open(os.path.join(path,'dbtest.pkl')))
    else:
        dbtest = None
    
    return dbtrain, dbval, dbtest

    print("Dataset Loaded in %d sec"%(time.time()-t))

    # }}}

if __name__ == "__main__":
    """
    This code can be tested with flickr_preprocess.ipynb to dig in what it does.
    """
    import pylab

    informations = json.load(open('./flickr30k/flickr30k/dataset.json'))
    informations = informations['images']
    img_path='./flickr30k/flickr30k/imgs/'

    splits = split_imgs(img_path, informations)
    train_imgs, val_imgs, test_imgs = splits['train_imgs'], splits['val_imgs'], splits['test_imgs']


    annotations=json.load(open(ann_path+'captions_%s2014.json'%split))['annotations']
    filename_to_raw_sentences = {info['filename']: [ sent['raw'] for sent in info['sentences']] for info in informations}
    assert(len(filename_to_raw_sentences) == len(train_imgs)+len(test_imgs)+len(val_imgs)), "Missing elements"
    
    print("End of main is not up to date, data was generated with ipynb. The required changes will be very small though.")
    # The main idea : use split images to get the right images depending on the split, loop_images() to prepare the actual  data, and dump.
    import pdb; pdb.set_trace()


    #print("Creating dbtrain ... \n")
    #dbtrain=loop_images(train_imgs,
    #                    annotations=filename_to_raw_sentences,
    #                    boxes_path=None)
    #print("Saving dbtrain.mat ... \n")
    #savemat(mdict=dbtrain,
    #        file_name='flickr30k/dbtrain.mat')
    ##pickle.dump(dbtrain,open('flick_data/dbtrain.pkl','wb'))
    #del dbtrain
