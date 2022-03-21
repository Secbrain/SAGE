import gc
import os
import traceback
import numpy as np
import tensorflow as tf
from scipy.misc import imsave,imresize
import bench_settings
from attacks.biased_boundary_attack import BiasedBoundaryAttack
from utils.distance_measures import DistL2
from models.blackbox_wrapper import BlackboxWrapper
from models.imagenet_inception_v3.foolbox_model import create_imagenet_iv3_model
from models.imagenet_irn_v2.foolbox_model import create_imagenet_irn_v2_model
from utils import dataset_imagenet
from utils.imagenet_labels import label_to_name
from utils.sampling.sampling_provider import SamplingProvider
from utils.util import line_search_to_boundary, find_closest_img
from scipy.io import loadmat
from scipy.ndimage import imread
import json

def main():
    n_classes = 1000
    img_shape = (299, 299, 3) #考虑先都是500，输入模型验证时在转到299!!!!!!!!!!
    imagenet_base_path = "/mnt/traffic/xzy/huawei/biased_boundary_attack/ILSVRC"
    #y_val = dataset_imagenet.load_dataset_y_val(imagenet_base_path, limit=None)
    rec=loadmat(os.path.join("/mnt/traffic/xzy/huawei/biased_boundary_attack/ILSVRC","devkit","data","ILSVRC2015_clsloc_validation_ground_truth.mat"))['rec'][0]
    img_lu='/mnt/traffic/xzy/wuxian/images/'
    
    #y = y_val
    #m = len(y)
    #indices = np.arange(m)
    jf=open('/mnt/traffic/xzy/wuxian/similar_images/yuanind.json') #给对应json文件!!!!!
    yuanind=json.load(jf)
    jf2=open('/mnt/traffic/xzy/wuxian/similar_images/target.json') #给近邻json文件!!!!!
    y_target=json.load(jf2)
    iid=np.loadtxt('/mnt/traffic/xzy/wuxian/similar_images/id.txt',dtype=np.int16)
    #np.random.seed(0)
    #y_target = np.random.randint(0, 1000, size=len(indices))

    with tf.Session() as sess:
        bb_model = BlackboxWrapper(create_imagenet_iv3_model(sess=sess))
        surr_model = create_imagenet_irn_v2_model(sess=sess) if bench_settings.USE_SURROGATE_BIAS else None
        dm_l2 = DistL2().to_range_255()
        with SamplingProvider(shape=img_shape, n_threads=3, queue_lengths=80) as sample_gen:
            with BiasedBoundaryAttack(bb_model, sample_gen, dm_main=dm_l2, substitute_model=surr_model) as bba_attack:
                n_calls_max = 500
                for i_img in range(2500):
                    i_img+=238
                    print(int(i_img))
                    clsid_gt=int(i_img/5)
                    img_orig=imread(img_lu+str(i_img)+'.jpg')
                    img_orig=imresize(img_orig,(299,299,3),interp='bilinear')
                    #img_orig = dataset_imagenet.load_on_demand_X_val(imagenet_base_path, [indices[i_img]])[0]
                    #clsid_gt = y[i_img]
                    clsid_target = iid[y_target[str(i_img)][0]]
                    #print("clsid_gt-------",clsid_gt)
                    #print("clsid_target------",clsid_target)
                    #print(indices[i_img])
                    d1=imread("/mnt/traffic/xzy/huawei/biased_boundary_attack/ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_"+str(yuanind[str(i_img)]).zfill(8)+".JPEG")
                    l1=len(rec[yuanind[str(i_img)]][0][0])
                    #print(rec[yuanind[i_img]][0][0])
                    y1=np.zeros(d1.shape)
                    for i2 in range(l1):
                        x1=rec[yuanind[str(i_img)]][0][0][i2][1][0]
                        y1[x1[1]:x1[3],x1[0]:x1[2]]=1
                        #print(x1[1],"--",x1[3],"--",x1[0],"--",x1[2])
                        #imsave("z1.jpg",y1)
                    y2=imresize(y1,img_shape)
                    #imsave("c1.jpg",y2)
                    print("Image {}, original clsid={} ({}), target clsid={} ({}):".format(i_img, clsid_gt, label_to_name(clsid_gt),
                                                                                           clsid_target, label_to_name(clsid_target)))

                    img_log_dir_final = os.path.join("out_imagenet_bench{}".format(bench_settings.EXPERIMENT_SUFFIX), "{}".format(i_img))
                    img_log_dir = img_log_dir_final + ".inprog"

                    if os.path.exists(img_log_dir_final) or os.path.exists(img_log_dir):
                        continue
                    try:
                        os.makedirs(img_log_dir, exist_ok=False)

                        bb_model.adv_set_target(orig_img=img_orig, is_targeted=False, label=clsid_gt, dist_measure=dm_l2,
                                                img_log_dir=img_log_dir, img_log_size=(299, 299), img_log_only_adv=True,
                                                print_calls_every=100) #改299成500,以及函数里的!!!!!!!

                        # Starting point: Pick the closest image of the target class.
                        #target_ids = np.arange(len(y_val))[y_val == clsid_target]
                        #X_targets = dataset_imagenet.load_on_demand_X_val(imagenet_base_path, indices=target_ids)

                        #X_start,xxid = find_closest_img(bb_model, X_orig=img_orig, X_targets=X_targets, label=clsid_target, is_targeted=True)

                        #print("------")
                        #print(xxid)
                        #print(target_ids[xxid])
                        #npyuan=np.array(X_targets)
                        #npxuan=np.array(X_start)
                        #print(np.argwhere(npyuan==npxuan))
                        #difang=np.argwhere(npyuan==npxuan)[0][0]
                        X_start=img_orig.copy()
                        X_start=X_start.astype(np.float32)
                        print('aaaaaaa')
                        for i in range(5):
                            difang=y_target[str(i_img)][i]#target_ids[xxid]
                            #print(difang)
                            #print(target_ids(difang))
                            e1=imread("/mnt/traffic/xzy/huawei/biased_boundary_attack/ILSVRC/Data/CLS-LOC/val/ILSVRC2012_val_"+str(difang).zfill(8)+".JPEG")
                            l2=len(rec[difang][0][0])
                            #print(rec[difang][0][0])
                            z1=np.zeros(e1.shape)
                            for i3 in range(l2):
                                x2=rec[difang][0][0][i3][1][0]
                                z1[x2[1]:x2[3],x2[0]:x2[2]]=1
                            z2=imresize(z1,img_shape)
                            z2[y2>z2]=y2[y2>z2]
                            #imsave("c2.jpg",z2)
                            e2=imresize(e1,img_shape)
                            #print(z2==1)
                            e2=e2.astype(np.float32)
                            #print(X_start.dtype)
                            #print(e2.dtype)
                            X_start[z2==255]+=0.2*(e2[z2==255]-X_start[z2==255])
                            if dm_l2.calc(X_start, img_orig.astype(np.float32))>23.:
                                break
                        print('bbbbbbbbb')
                        X_start=X_start.astype(np.int16)
                        imsave(os.path.join(img_log_dir, "xstart.png"), X_start)
                        #imsave(os.path.join(img_log_dir, "c1.png"), y2)
                        #imsave(os.path.join(img_log_dir, "c2.png"), z2)
                        #z2[y2>z2]=y2[y2>z2]
                        #imsave(os.path.join(img_log_dir, "cbing.png"), z2)

                        X_start = line_search_to_boundary(bb_model, x_orig=img_orig, x_start=X_start, label=clsid_gt, is_targeted=False)
                        bba_attack.run_attack(img_orig, label=clsid_gt, is_targeted=False, X_start=X_start,
                                              n_calls_left_fn=(lambda: n_calls_max - bb_model.adv_get_n_calls()),
                                              source_step=2e-3, spherical_step=5e-2,
                                              mask=None, recalc_mask_every=(1 if bench_settings.USE_MASK_BIAS else None))

                        print("cishu-------------",bb_model.adv_get_n_calls())
                        final_adversarial = bb_model.adv_get_best_img()

                        #imsave(os.path.join(img_log_dir, "c1.png"), y2)
                        #imsave(os.path.join(img_log_dir, "c2.png"), z2)
                        #imsave(os.path.join(img_log_dir, "xstart-1.png"), X_strat)
                        imsave(os.path.join(img_log_dir, "clean.png"), np.uint8(img_orig))
                        imsave(os.path.join(img_log_dir, "ae-final.png"), np.uint8(imresize(final_adversarial,(500,500,3))))
                        diff = np.float32(final_adversarial) - np.float32(img_orig) + 127.
                        imsave(os.path.join(img_log_dir, "diff.png"), np.uint8(np.round(diff)))
                        diff2= np.float32(final_adversarial) - np.float32(img_orig)
                        imsave(os.path.join(img_log_dir, "diffyuan.png"), np.uint8(np.round(diff2)))
                        os.rename(img_log_dir, img_log_dir_final)
                        gc.collect()
                    except Exception:
                        # Log, but keep going.
                        print("Error trying to find adversarial example!")
                        traceback.print_exc()


if __name__ == "__main__":
    main()
