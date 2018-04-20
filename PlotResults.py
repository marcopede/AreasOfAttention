#plot results over different iterations

try:
    import subprocess, os
    gpu_id = subprocess.check_output('gpu_getIDs.sh', shell=True)
    os.environ["THEANO_FLAGS"]='device=gpu%s'%gpu_id
    print(os.environ["THEANO_FLAGS"])
except:
    pass


#import argparse

import numpy
import bcolors
import os
import sys
import cPickle as pickle
import numpy as np

from CaptionEvaluation import cmd_line_parser, buildNetwork, setParamNetwork, evaluateCaptions, generateCaptions, load_coco, compileNetwork
from CaptionEvaluation import evaluateCaptionsFlickr
from flickr_prepareData import load_flickr

def early_stopping(filepath, save=False):
    if not os.path.isfile(filepath):
        print("Invalid path to .res: " + filepath )
    else:
        all_res = pickle.load(open(filepath))

        all_best_val_iter_dict = dict()
        for beam_size in all_best_val_iter_dict.keys():
            print("Beam size: " + str(beam_size))
            res = all_res[key]
            for key in res.keys():
                print(key + ":")
                local_max = -1
                local_max_iter = -1
                iteration = 0
                for value in res[key]:
                    iteration = iteration + 1
                    if value > local_max:
                        local_max = value
                        local_max_iter = iteration
                best_val_iter_dict[key] = (local_max,local_max_iter)
                print("Best value: " + str(local_max) + " at iteration " + str(local_max_iter))
            all_best_val_iter_dict[beam_size] = best_val_iter_dict
    print(all_best_val_iter_dict)
    return all_best_val_iter_dict

def plot_curves(res, plot_name=None):
    import pylab
    pylab.figure()
    pylab.clf()
    for c in res:
        pylab.plot(range(1,len(res[c])+1),res[c],label=c)
    pylab.grid()
    pylab.legend(loc=9,fontsize='x-small')

    pylab.draw()
    if params['save_plot']:
        assert(not plot_name is None)
        pylab.savefig(plot_name)

    if params['display']:
        pylab.show()



if __name__ == "__main__":
    #parse command line
    parser = cmd_line_parser()
    parser.add_argument('--recompute', action='store_true', help='Recompute the curves, even if the saved file is present!')
    parser.add_argument('--add', action='store_true', help='Add the missing checkpoints without recomputing the previous ones!')
    parser.add_argument('--res_folder',dest='res_folder', default='', help='results will saved in given results folder.')
    #parser.add_argument('--start_iter', dest='start_iter', type=int, default=-1, help='Number of the first snapshot to be evaluated')
    #parser.add_argument('--end_iter', dest='end_iter', type=int, default=-1, help='Number of the last snapshot to be evaluated')
    parser.add_argument('--skip_process_results', action='store_true', help='If you do not want the results to be processed and printed in console.')
    parser.add_argument('--show_sent', action='store_true', help='Print the captions while evaluating')
    parser.add_argument('--display', action='store_true', help='Display a plot of the scores againt iterations')
    parser.add_argument('--save_plot', action='store_true', help='Save an export of the plot (or over-ride it if it exists)')
    parser.add_argument('--partial_save', type=int, default=0, help='Will save the results after each evaluation. Should only be used when launching in best effort and with --add')

    ### Following option not handled yet (will simply print a message for now)
    parser.add_argument('--process_res_savepath', type=str, default='', help='Path to save the processed results of early stopping')
    #parser.add_argument('--imgfeedback_mechanism', type=str, default='simple', help='Choose the mechanism to generate the density tempering parameter. Choices: "simple", "highres"')


    #parser.add_argument('-b', '--beam_size', dest='beam_size', type=int, default=1, help='Size of the beam in the beam search')
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict

    if params['res_folder'] != '':
        model_name = params['filename'].split("/")[-2]
        pkl_files_name = params['filename'].split("/")[-1]
        results_directory = os.path.join(params['res_folder'], model_name)

        # Name of the file where we store raw results
        results_file = os.path.join(results_directory, "".join([pkl_files_name,".res"]))
        plot_name = os.path.join(results_directory, "".join([pkl_files_name,"_plot.pdf"]))
         #= pkl_files_name + "_plot" # Used only if --save_plot

        if params['conv_reduced']:
            results_file = results_file+"_convreduced"
        #results_file = os.path.join(params['res_folder'], model_name, "".join([pkl_files_name,".res"]))

    else:
        results_file = params['filename']+'.res'
        if params['conv_reduced']:
            results_file = results_file+"_convreduced"
        #plot_name = os.path.join(results_directory, "".join([pkl_files_name,"_plot.pdf"]))

    # Name of the file where we store processed results
    processed_results_file = results_file+"processed"

    #work_done =




    if os.path.isfile(results_file) and not (params['recompute'] or params['add']):
        from bcolors import bcolors
        print(bcolors.FAIL + "There is already a file present. You must use either option 'add' or 'recompute', even if you "+ \
              "are asking for a new beamsize" + bcolors.ENDC)
        sys.exit()

    if not os.path.isfile(results_file) or params['recompute'] or params['add']:
    #if not os.path.isfile(params['filename']+'.res') or params['recompute'] or params['add']:

        import glob
        checkpoints = glob.glob(params['filename']+"*.pkl")
        sorted_checkpoints = []
        #for i in range(len(checkpoints)+1):
        for i in range(501):
            for element in checkpoints:
                number = int(element.split("_")[-1].split(".")[0])
                if number==i:
                    sorted_checkpoints.append(element)
        checkpoints = sorted_checkpoints

        print('Available checkpoints:',checkpoints)


        d = pickle.load(open(checkpoints[0]))
        vocab = d['vocab']
        word_to_index = d['word_to_index']
        index_to_word = d['index_to_word']
        CFG=d['config']
        CFG['SKIP_PROCESS_RESULTS'] = params['skip_process_results']
        CFG['DATASET'] = params['dataset']
        #overwrite some parameters
        if params['add']:
            if os.path.isfile(results_file):
                all_results = pickle.load(open(results_file))
                if (str(params['beam_size']) in all_results.keys()) and (len(all_results[str(params['beam_size'])]) > 0):
                    new_beam = False
                    res = all_results[str(params['beam_size'])]
                    #res = pickle.load(results_file)[str(params['beam_size'])]
                    print('Loading precomputed values. If you want to recompute them use the option --recompute')
                    #res = pickle.load(open(params['filename']+'.res'))

                    start_from = len(res['CIDEr'])
                    if len(checkpoints)>start_from:
                        print('Starting from '+checkpoints[start_from])
                        checkpoints = checkpoints[start_from:]
                    else:
                        print('No new checkpoints to evalaute for this beam size')
                        if params['display'] or params['save_plot']:
                                plot_curves(res, plot_name)

                        #if not CFG['SKIP_PROCESS_RESULTS']:
                        #    save = False
                        #    filepath = results_file
                        #    all_best_val_iter_dict = early_stopping(filepath, save)
                        #    for beam_size in all_best_val_iter_dict.keys():
                        #        print("Beam size: " + str(beam_size))
                        #        best_val_iter_dict = all_best_val_iter_dict[beam_size]
                        #        for key in best_val_iter_dict.keys():
                        #            print(key + ": " + str(best_val_iter_dict[key]))

                        #    pickle.dump(best_val_iter_dict,open(processed_results_file,'w+'))

                        sys.exit()

                else:
                    res = dict()
                    new_beam=True
            else:
                res = dict()
                all_results = dict()
                new_beam=False
        else:
            all_results = dict()
            res = dict()
            new_beam=False

        net = buildNetwork(CFG,params,vocab)
        setParamNetwork(CFG,net['out'],net['cnn'],net['sent_emb'],net['out_reg'],d)
        f_cnn, emb_cnn, f, f_sent = compileNetwork(CFG,net)

        if params['dataset'] == 'coco':
            dbval,dbtest = load_coco(no_train=True,add_validation=CFG['ADD_VALIDATION'])
            if CFG['USE_TEST_SPLIT']:
                dbval=dbtest
        elif params['dataset'] == 'flickr':
            _, dbval, dbtest = load_flickr(no_train=True, no_test=False)
            #dbval, _, dbtest = load_flickr(no_train=False, no_test=False)

        import os
        if not os.path.exists('./temp'):
            os.makedirs('./temp')
        import uuid
        captionfile = './temp/'+str(uuid.uuid4())+".json"

        results = []
        for ch in checkpoints:
            evaluate_ch = True
            #if (params['start_iter'] != -1) or (params['end_iter'] != -1):
            #    iteration = int(ch.split(".pkl")[0].split("_")[-1])
            #    _end = max(1,params['end_iter'])
            #    _start = max(1,params['start_iter'])
            #    if iteration < _start or iteration > _end:
            #        evaluate_ch = False

            if evaluate_ch:
                print('Generating Captions for {} ...'.format(ch))
                d = pickle.load(open(ch))
                #vocab = d['vocab']
                #word_to_index = d['word_to_index']
                #index_to_word = d['index_to_word']
                #CFG=d['config']
                setParamNetwork(CFG,net['out'],net['cnn'],net['sent_emb'],net['out_reg'],d,show_layers=False)
                show_sent = params['show_sent']
                generateCaptions(captionfile,dbval,word_to_index,index_to_word,CFG,params,f_cnn,emb_cnn,f,f_sent,VISUALIZE=params['visualize'],BEAM_SIZE=params['beam_size'],show_sent=show_sent)

                print("We are about to call evaluation on generated captions...")
                if params['dataset'] == "coco":
                    results.append(evaluateCaptions(captionfile))
                elif params['dataset'] == "flickr":
                    #results.append(evaluateCaptions(captionfile))
                    # What we want to do in the end
                    results.append(evaluateCaptionsFlickr(captionfile))


            if params['partial_save']:
                if (not params['add']) or new_beam or (len(res.keys())==0):
                    res = results[0].copy()
                    for c in res:
                        res[c]=[]
                    new_beam=False
                for p in results:
                    for c in res:
                        res[c].append(p[c])
                import os
                if params['res_folder'] != '':
                    if not os.path.exists(results_directory):
                        os.makedirs(results_directory)
                res_dict = all_results
                res_dict[str(params['beam_size'])] = res

                pickle.dump(res_dict,open(results_file,'w+'))
                os.remove(captionfile)
                results = []


        #convert a list of dict into a dict of lists
        if (not params['add']) or new_beam or (len(res.keys())==0):
            res = results[0].copy()
            for c in res:
                res[c]=[]
        for p in results:
            for c in res:
                res[c].append(p[c])
        #pickle.dump(res,open(results_file,'wb'))
        import os
        if params['res_folder'] != '':
            if not os.path.exists(results_directory):
                os.makedirs(results_directory)

        res_dict = all_results
        res_dict[str(params['beam_size'])] = res
        pickle.dump(res_dict,open(results_file,'w+'))
        try:
            os.remove(captionfile)
        except:
            print("File already deleted")
    else:
        print('Loading precomputed values. If you want to recompute them use the option --recompute')
        #res = pickle.load(open(params['filename']+'.res'))
        all_res = pickle.load(open(results_file))
        res = all_res[str(params['beam_size'])]

    if params['display'] or params['save_plot']:
            plot_curves(res, plot_name)
    print("Ok, done for "+params['filename'])
    ###
    #if not CFG['SKIP_PROCESS_RESULTS']:
    #    save = False
    #    filepath = results_file
    #    all_best_val_iter_dict = early_stopping(filepath, save)
    #    for beam_size in all_best_val_iter_dict.keys():
    #        print("Beam size: " + str(beam_size))
    #        best_val_iter_dict = all_best_val_iter_dict[beam_size]
    #        for key in best_val_iter_dict.keys():
    #            print(key + ": " + str(best_val_iter_dict[key]))

    #    pickle.dump(best_val_iter_dict,open(processed_results_file,'w+'))
