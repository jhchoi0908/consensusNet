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
	path_test	= args.test_dir
	ckpt_dir	= args.ckpt_dir
	random_value	= 19201
	
	init_denoisers	= ['rednet10', 'rednet20', 'rednet30', 'rednet40', 'rednet50']
	model_0		= [network0(os.path.join(args.ckpt_dir, den)) for den in init_denoisers]
	model_1		= network1(os.path.join(args.ckpt_dir, 'mse_estimator'))
	model_2		= network2(os.path.join(args.ckpt_dir, 'booster_T3'))
	model_blind30	= network0(os.path.join(args.ckpt_dir, 'rednet_blind30'))
	model_blind150	= network0(os.path.join(args.ckpt_dir, 'rednet_blind'))
	
	np.random.seed(random_value)
	files		= os.listdir(path_test)
	sigmaSet	= range(10, 51)
	PSNR_to		= np.zeros((len(sigmaSet), len(init_denoisers)+9))
	
	for sigma in range(10, 51, 5):
		path_gt		= os.path.join(args.out_dir, str(sigma), 'groundtruth')
		path_in		= os.path.join(args.out_dir, str(sigma), 'inputs')
		path_10		= os.path.join(args.out_dir, str(sigma), 'rednet10')
		path_20		= os.path.join(args.out_dir, str(sigma), 'rednet20')
		path_30		= os.path.join(args.out_dir, str(sigma), 'rednet30')
		path_40		= os.path.join(args.out_dir, str(sigma), 'rednet40')
		path_50		= os.path.join(args.out_dir, str(sigma), 'rednet50')
		path_before	= os.path.join(args.out_dir, str(sigma), 'before')
		path_after	= os.path.join(args.out_dir, str(sigma), 'after')
		path_blind30	= os.path.join(args.out_dir, str(sigma), 'bline30')
		path_blind150	= os.path.join(args.out_dir, str(sigma), 'blind150')
		
		if not os.path.exists(path_gt):
			os.makedirs(path_gt)
		if not os.path.exists(path_in):
			os.makedirs(path_in)
		if not os.path.exists(path_10):
			os.makedirs(path_10)
		if not os.path.exists(path_20):
			os.makedirs(path_20)
		if not os.path.exists(path_30):
			os.makedirs(path_30)
		if not os.path.exists(path_40):
			os.makedirs(path_40)
		if not os.path.exists(path_50):
			os.makedirs(path_50)
		if not os.path.exists(path_before):
			os.makedirs(path_before)
		if not os.path.exists(path_after):
			os.makedirs(path_after)
		if not os.path.exists(path_blind30):
			os.makedirs(path_blind30)
		if not os.path.exists(path_blind150):
			os.makedirs(path_blind150)
	
	for fname in files:
		
		print("[*] %s"%fname)
		img_gt	= load_image(os.path.join(path_test, fname))
		
		for i, sigma in enumerate(sigmaSet):
			sigma_f	= sigma/255.0
			img_no	= img_gt + sigma_f*np.random.normal(size=img_gt.shape)
			img_no4	= np.reshape(img_no, (1, img_no.shape[0], img_no.shape[1], 1))
			img_de	= [model.run(img_no4)[0,:,:,0] for model in model_0]
			
			# NN
			mse_est	= [model_1.run(extract_patches(img_no, patch_size), extract_patches(den, patch_size)) for den in img_de]
			img_com_nn, _	= combine(img_de, mse_est)
			img_out_nn	= model_2.run(img_no4, np.reshape(img_com_nn, (1, img_com_nn.shape[0], img_com_nn.shape[1], 1)))[0,:,:,0]
			
			# SURE
			sz	= img_no.shape
			N	= sz[0]*sz[1]
			b	= np.random.randn(1, sz[0], sz[1], 1)
			Eps	= sigma_f/100.0
			img_no4_= img_no4 + Eps*b
			img_de_	= [model.run(img_no4_)[0,:,:,0] for model in model_0]
			SURE	= [cal_SURE(img_no, img_de[j], img_de_[j], sigma_f, b, Eps, N) for j in range(len(img_de))]
			img_com_sure, _	= combine(img_de, SURE)
			img_out_sure	= model_2.run(img_no4, np.reshape(img_com_sure, (1, img_com_sure.shape[0], img_com_sure.shape[1], 1)))[0,:,:,0]
			
			# Oracle
			mse_tru	= [cal_mse(den, img_gt) for den in img_de]
			img_com_ora, _	= combine(img_de, mse_tru)
			img_out_ora	= model_2.run(img_no4, np.reshape(img_com_ora, (1, img_com_ora.shape[0], img_com_ora.shape[1], 1)))[0,:,:,0]
			
			# REDNet Blind
			img_den_b30	= model_blind30.run(img_no4)[0,:,:,0]
			img_den_b150	= model_blind150.run(img_no4)[0,:,:,0]
			
			PSNR	= [cal_psnr(img_gt, img_no)]		+ [cal_psnr(img_gt, temp) for temp in img_de]	\
				+ [cal_psnr(img_gt, img_com_nn)]	+ [cal_psnr(img_gt, img_out_nn)]	+ [cal_psnr(img_gt, img_com_sure)]	+ [cal_psnr(img_gt, img_out_sure)]	\
				+ [cal_psnr(img_gt, img_com_ora)]	+ [cal_psnr(img_gt, img_out_ora)]	+ [cal_psnr(img_gt, img_den_b30)]	+ [cal_psnr(img_gt, img_den_b150)]
			
			PSNR_to[i, :]	+= PSNR
			print("  [sigma %2d] %.4f   %.4f %.4f %.4f %.4f %.4f   %.4f %.4f   %.4f %.4f   %.4f %.4f   %.4f %.4f" % \
				(sigma, PSNR[0], PSNR[1], PSNR[2], PSNR[3], PSNR[4], PSNR[5], PSNR[6], PSNR[7], PSNR[8], PSNR[9], PSNR[10], PSNR[11], PSNR[12], PSNR[13]))
			
			if sigma%5==0:
				path_gt		= os.path.join(args.out_dir, str(sigma), 'groundtruth')
				path_in		= os.path.join(args.out_dir, str(sigma), 'inputs')
				path_10		= os.path.join(args.out_dir, str(sigma), 'rednet10')
				path_20		= os.path.join(args.out_dir, str(sigma), 'rednet20')
				path_30		= os.path.join(args.out_dir, str(sigma), 'rednet30')
				path_40		= os.path.join(args.out_dir, str(sigma), 'rednet40')
				path_50		= os.path.join(args.out_dir, str(sigma), 'rednet50')
				path_before	= os.path.join(args.out_dir, str(sigma), 'before')
				path_after	= os.path.join(args.out_dir, str(sigma), 'after')
				path_blind30	= os.path.join(args.out_dir, str(sigma), 'bline30')
				path_blind150	= os.path.join(args.out_dir, str(sigma), 'blind150')
				
				save_image(img_gt,		os.path.join(path_gt, fname))
				save_image(img_no,		os.path.join(path_in, fname))
				np.savetxt(os.path.join(path_in, fname[:-4]+'.txt'),	img_no)
				save_image(img_de[0],		os.path.join(path_10, fname))
				save_image(img_de[1],		os.path.join(path_20, fname))
				save_image(img_de[2],		os.path.join(path_30, fname))
				save_image(img_de[3],		os.path.join(path_40, fname))
				save_image(img_de[4],		os.path.join(path_50, fname))
				save_image(img_com_nn,		os.path.join(path_before, fname))
				save_image(img_out_nn,		os.path.join(path_after,  fname))
				save_image(img_den_b30,		os.path.join(path_blind30, fname))
				save_image(img_den_b150,	os.path.join(path_blind150, fname))
	
	PSNR_to	/= len(files)
	print("[*] final")
	for i, sigma in enumerate(sigmaSet):
		print("  [sigma %2d] %.4f   %.4f %.4f %.4f %.4f %.4f   %.4f %.4f   %.4f %.4f   %.4f %.4f   %.4f %.4f" % \
			(sigma, PSNR_to[i,0], PSNR_to[i,1], PSNR_to[i,2], PSNR_to[i,3], PSNR_to[i,4], PSNR_to[i,5], PSNR_to[i,6], PSNR_to[i,7], PSNR_to[i,8], \
			PSNR_to[i,9], PSNR_to[i,10], PSNR_to[i,11], PSNR_to[i,12], PSNR_to[i,13]))
	

if __name__ == "__main__":
	main()

