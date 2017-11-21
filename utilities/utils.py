import tensorflow	as tf
import numpy		as np
import math, os, random
from scipy.ndimage	import imread
from scipy.misc		import toimage

def get_files(directory, israndom=False):
	
	files	= []
	for f in os.listdir(directory):
		if f.lower().endswith(".jpg") or f.lower().endswith(".jpeg") or f.lower().endswith(".png"):
			image	= imread(os.path.join(directory, f), flatten=True)/255.0
			files.append(f)
	if israndom:
		random.shuffle(files)
	else:
		files.sort()
	return files


def get_common_files(directory1, directory2):
	files	= [f for f in os.listdir(directory1) if (f.lower().endswith(".jpg") or f.lower().endswith(".jpeg") or f.lower().endswith(".png")) and (f in os.listdir(directory2))]
	files.sort()
	return files


def load_data(directory, files, patch_sz, num_patches):
	size	= num_patches*len(files)
	data	= np.empty([size, patch_sz, patch_sz, 1])
	cnt	= 0
	for fname in files:
		image	= imread(os.path.join(directory, fname), flatten=True)/255.0
		[height, width]	= image.shape
		if height>patch_sz and width>patch_sz:
			rows	= np.random.randint(height-patch_sz+1, size=num_patches)
			cols	= np.random.randint(width -patch_sz+1, size=num_patches)
			for i in range(num_patches):
				data[cnt,:,:,0]	= image[rows[i]:rows[i]+patch_sz, cols[i]:cols[i]+patch_sz]
				cnt += 1
				if cnt%10000==0:
					print("%d batches"%cnt)
	print("Total %d batches"%cnt)
	print data.shape
	data	= data[:cnt]
	shuf	= np.arange(len(data))
	np.random.shuffle(shuf)
	data	= data[shuf]
	return data


def load_image(directory, filename, grayscale=True):
	img	= imread(os.path.join(directory, filename), flatten=grayscale)
	img	= np.round(img).astype(np.float32)/255.0
	return img


def save_image(image, filename):
	toimage(image, cmin=0.0, cmax=1.0).save(filename)


def add_noise(data, sigma):
	noisy	= data + np.random.normal(scale=sigma/255.0, size=data.shape)
	return np.float32(noisy)


def cal_std(p):
	return np.std(np.array(p))


def cal_psnr(x, y):
	return 10*np.log10(1/np.mean(np.square(x-y)))


def cal_mse(x, y):
	return np.mean(np.square(x-y))


def cal_estimated_mse(y, model, sigma):
	
	N	= float(y.shape[1]*y.shape[2])
	epsilon	= sigma/100.0/255.0
	b	= np.random.normal(size=y.shape)
	z	= y + epsilon*b
	
	y_	= model.run(y)
	z_	= model.run(z)
	div	= (1/epsilon)*np.sum(np.multiply(b, z_-y_))
	MSE	= np.mean(np.square(y - y_)) - (sigma/255.0)**2 + 2*(sigma/255.0)**2*div/N
	return MSE, y_


def weighted_average(y, models, trained_sigmas, sigma):
	
	c_r	= 0.3
	c_s	= 0.3
	
	y_	= [None]*len(models)
	mse_	= [None]*len(models)
	for i in range(len(models)):
		mse_[i], y_[i]	= cal_estimated_mse(y, models[i], sigma)
	
	mse_	= [msei-min(mse_) for msei in mse_]
	sigma_r	= cal_std(mse_)
	weight1	= [math.exp(-msei/(c_r*sigma_r)) for msei in mse_]
	
	sigma_	= [float(s-sigma) for s in trained_sigmas]
	sigma_s	= max(cal_std(sigma_), 1e-3)
	weight2	= [math.exp(-abs(sigmai)/(c_s*sigma_s)) for sigmai in sigma_]
	
	weights	= [w1*w2 for w1, w2 in zip(weight1, weight2)]
	weights	= [wi/sum(weights) for wi in weights]
	output	= sum(yi*wi for yi, wi in zip(y_, weights))
	
	return output

