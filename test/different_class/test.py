import os, sys, time, argparse, random
import tensorflow	as tf
import numpy		as np
sys.path.append('../../')
from utils		import *
from network		import *

parser	= argparse.ArgumentParser(description='')
parser.add_argument('--patch_size',	dest='patch_size',	type=int,	default=64,				help='patch size')
parser.add_argument('--test_dir',	dest='test_dir',			default='../../data',			help='the directory for validation data')
parser.add_argument('--ckpt_dir',	dest='ckpt_dir',			default='../../trained_model',		help='the directory for meta file')
parser.add_argument('--out_dir',	dest='out_dir',				default='outputs',			help='the directory for output file')
args	= parser.parse_args()


def main():
	
	start_time_tot	= time.time()
	
	patch_size	= args.patch_size
	ckpt_dir	= args.ckpt_dir
	random_value	= 13928
	
	sigma		= 20
	init_denoisers	= ['building', 'face', 'flower']
	model_0		= [network_class(os.path.join(args.ckpt_dir, 'rednet_'+den+'_'+str(sigma))) for den in init_denoisers]
	model_1		= network1(os.path.join(args.ckpt_dir, 'mse_estimator'))
	model_2		= network2(os.path.join(args.ckpt_dir, 'booster_class'))
	model_rednet	= network0(os.path.join(args.ckpt_dir, 'rednet20'))
	
	np.random.seed(random_value)
	PSNR_to		= np.zeros((len(init_denoisers), len(init_denoisers)+6))
	
	for i, den in enumerate(init_denoisers):
		path_gt		= os.path.join(args.out_dir, den, 'groundtruth')
		path_in		= os.path.join(args.out_dir, den, 'inputs')
		path_building	= os.path.join(args.out_dir, den, 'building')
		path_face	= os.path.join(args.out_dir, den, 'face')
		path_flower	= os.path.join(args.out_dir, den, 'flower')
		path_before	= os.path.join(args.out_dir, den, 'before')
		path_after	= os.path.join(args.out_dir, den, 'after')
		path_rednet	= os.path.join(args.out_dir, den, 'rednet_generic')
		if not os.path.exists(path_gt):
			os.makedirs(path_gt)
		if not os.path.exists(path_in):
			os.makedirs(path_in)
		if not os.path.exists(path_building):
			os.makedirs(path_building)
		if not os.path.exists(path_face):
			os.makedirs(path_face)
		if not os.path.exists(path_flower):
			os.makedirs(path_flower)
		if not os.path.exists(path_before):
			os.makedirs(path_before)
		if not os.path.exists(path_after):
			os.makedirs(path_after)
		if not os.path.exists(path_rednet):
			os.makedirs(path_rednet)
		
		path_test	= os.path.join(args.test_dir, den)
		files		= sorted(os.listdir(path_test))
		print("[*] %s"%den)
		for fname in files:
			img_gt	= load_image(os.path.join(path_test, fname))
			sigma_f	= sigma/255.0
			img_no	= img_gt + sigma_f*np.random.normal(size=img_gt.shape)
			img_no4	= np.reshape(img_no, (1, img_no.shape[0], img_no.shape[1], 1))
			img_de	= [model.run(img_no4)[0,:,:,0] for model in model_0]
			
			# NN
			mse_est	= [model_1.run(extract_patches(img_no, patch_size), extract_patches(den, patch_size)) for den in img_de]
			img_com_nn, _	= combine(img_de, mse_est)
			img_out_nn	= model_2.run(img_no4, np.reshape(img_com_nn, (1, img_com_nn.shape[0], img_com_nn.shape[1], 1)))[0,:,:,0]
			
			# Oracle
			mse_tru	= [cal_mse(den, img_gt) for den in img_de]
			img_com_ora, _	= combine(img_de, mse_tru)
			img_out_ora	= model_2.run(img_no4, np.reshape(img_com_ora, (1, img_com_ora.shape[0], img_com_ora.shape[1], 1)))[0,:,:,0]
			
			# REDNet Generic
			img_gen	= model_rednet.run(img_no4)[0,:,:,0]
			
			PSNR	= [cal_psnr(img_gt, img_no)]		+ [cal_psnr(img_gt, temp) for temp in img_de]	\
				+ [cal_psnr(img_gt, img_com_nn)]	+ [cal_psnr(img_gt, img_out_nn)]	\
				+ [cal_psnr(img_gt, img_com_ora)]	+ [cal_psnr(img_gt, img_out_ora)]	+ [cal_psnr(img_gt, img_gen)]
			PSNR_to[i, :]	+= PSNR
			print("  [%15s] %.4f   %.4f %.4f %.4f   %.4f %.4f   %.4f %.4f   %.4f" % (fname, PSNR[0], PSNR[1], PSNR[2], PSNR[3], PSNR[4], PSNR[5], PSNR[6], PSNR[7], PSNR[8]))
			
			save_image(img_gt,		os.path.join(path_gt, fname))
			save_image(img_no,		os.path.join(path_in, fname))
			np.savetxt(os.path.join(path_in, fname[:-4]+'.txt'),	img_no)
			save_image(img_de[0],		os.path.join(path_building, fname))
			save_image(img_de[1],		os.path.join(path_face, fname))
			save_image(img_de[2],		os.path.join(path_flower, fname))
			save_image(img_com_nn,		os.path.join(path_before, fname))
			save_image(img_out_nn,		os.path.join(path_after,  fname))
			save_image(img_gen,		os.path.join(path_rednet, fname))
		
		PSNR_to[i,:]	/= len(files)
	
	print("[*] final")
	for i, den in enumerate(init_denoisers):
		print("  [class %10s] %.4f   %.4f %.4f %.4f   %.4f %.4f   %.4f %.4f   %.4f" % \
			(den, PSNR_to[i,0], PSNR_to[i,1], PSNR_to[i,2], PSNR_to[i,3], PSNR_to[i,4], PSNR_to[i,5], PSNR_to[i,6], PSNR_to[i,7], PSNR_to[i,8]))
	print("Total Time: %.4f" % (time.time()-start_time_tot))


if __name__ == "__main__":
	main()

