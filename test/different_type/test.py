import os, sys, time, argparse, random
import tensorflow	as tf
import numpy		as np
sys.path.append('../../')
from utils		import *
from network		import *

parser	= argparse.ArgumentParser(description='')
parser.add_argument('--patch_size',	dest='patch_size',	type=int,	default=64,				help='patch size')
parser.add_argument('--test_dir',	dest='test_dir',			default='../../',			help='the directory for validation data')
parser.add_argument('--ckpt_dir',	dest='ckpt_dir',			default='../../trained_model',		help='the directory for meta file')
args	= parser.parse_args()


def main():
	
	patch_size	= args.patch_size
	ckpt_dir	= args.ckpt_dir
	sigmaSet	= range(10, 51, 10)
	PSNR_to		= np.zeros((len(sigmaSet), 11))
	
	for i, sigma in enumerate(sigmaSet):
		
		start_time_tot	= time.time()
		
		test_dir	= os.path.join(args.test_dir, str(sigma))
		path_gt		= os.path.join(test_dir, 'groundtruth')
		path_in		= os.path.join(test_dir, 'inputs')
		path_dncnn	= os.path.join(test_dir, 'dncnn')
		path_ffdnet	= os.path.join(test_dir, 'ffdnet')
		path_bm3d	= os.path.join(test_dir, 'bm3d')
		path_rednet	= os.path.join(test_dir, 'rednet')
		path_den	= [path_dncnn, path_ffdnet, path_bm3d, path_rednet]
		path_before	= os.path.join(test_dir, 'before')
		path_after	= os.path.join(test_dir, 'after')
		
		if not os.path.exists(path_before):
			os.makedirs(path_before)
		if not os.path.exists(path_after):
			os.makedirs(path_after)
		
		model_rednet	= network0(os.path.join(args.ckpt_dir, 'rednet'+str(sigma)))
		
		model_1		= network1(os.path.join(args.ckpt_dir, 'mse_estimator'))
		model_2		= network2(os.path.join(args.ckpt_dir, 'booster_type'))
		model_blind30	= network0(os.path.join(args.ckpt_dir, 'rednet_blind30'))
		model_blind150	= network0(os.path.join(args.ckpt_dir, 'rednet_blind'))
		
		files		= os.listdir(path_gt)
		
		for fname in files:
			img_gt	= load_image(os.path.join(path_gt, fname))
			img_no	= np.loadtxt(os.path.join(path_in, fname[:-4]+'.txt'), delimiter=' ')
			img_no4	= np.reshape(img_no, (1, img_no.shape[0], img_no.shape[1], 1))
			img_de	= [load_image(os.path.join(path, fname)) for path in path_den]
			
			# NN
			mse_est	= [model_1.run(extract_patches(img_no, patch_size), extract_patches(den, patch_size)) for den in img_de]
			img_com_nn, _	= combine(img_de, mse_est)
			img_out_nn	= model_2.run(img_no4, np.reshape(img_com_nn, (1, img_com_nn.shape[0], img_com_nn.shape[1], 1)))[0,:,:,0]
			
			# Oracle
			mse_tru	= [cal_mse(den, img_gt) for den in img_de]
			img_com_ora, _	= combine(img_de, mse_tru)
			img_out_ora	= model_2.run(img_no4, np.reshape(img_com_ora, (1, img_com_ora.shape[0], img_com_ora.shape[1], 1)))[0,:,:,0]
			
			# REDNet Blind
			img_den_b30	= model_blind30.run(img_no4)[0,:,:,0]
			img_den_b150	= model_blind150.run(img_no4)[0,:,:,0]
			
			PSNR	= [cal_psnr(img_gt, img_no)]		+ [cal_psnr(img_gt, temp) for temp in img_de]	\
				+ [cal_psnr(img_gt, img_com_nn)]	+ [cal_psnr(img_gt, img_out_nn)]	\
				+ [cal_psnr(img_gt, img_com_ora)]	+ [cal_psnr(img_gt, img_out_ora)]	\
				+ [cal_psnr(img_gt, img_den_b30)]	+ [cal_psnr(img_gt, img_den_b150)]
			for j in range(len(PSNR)):
				PSNR_to[i, j]	+= PSNR[j]
			print("  [sigma %2d] %.4f   %.4f %.4f %.4f %.4f   %.4f %.4f   %.4f %.4f   %.4f %.4f" % \
				(sigma, PSNR[0], PSNR[1], PSNR[2], PSNR[3], PSNR[4], PSNR[5], PSNR[6], PSNR[7], PSNR[8], PSNR[9], PSNR[10]))
			
			save_image(img_com_nn, os.path.join(path_before, fname))
			save_image(img_out_nn, os.path.join(path_after,  fname))
	
	PSNR_to	/= len(files)
	print("[*] final")
	for i, sigma in enumerate(sigmaSet):
		print("  [sigma %2d] %.4f   %.4f %.4f %.4f %.4f   %.4f %.4f   %.4f %.4f   %.4f %.4f" % \
			(sigma, PSNR_to[i,0], PSNR_to[i,1], PSNR_to[i,2], PSNR_to[i,3], PSNR_to[i,4], PSNR_to[i,5], PSNR_to[i,6], PSNR_to[i,7], PSNR_to[i,8], PSNR_to[i,9], PSNR_to[i,10]))
	

if __name__ == "__main__":
	main()

