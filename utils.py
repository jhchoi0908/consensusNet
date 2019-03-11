import numpy		as np
from scipy.ndimage	import imread
from scipy.misc		import toimage
import sys, random
import cvxpy		as cvx

def extract_patches_random(img, patch_size, num_patches):
	
	[height, width]	= img.shape
	
	rows	= np.random.randint(height-patch_size+1, size=num_patches)
	cols	= np.random.randint(width -patch_size+1, size=num_patches)
	
	data	= np.empty([num_patches, patch_size, patch_size, 1])
	cnt	= 0
	for r, c in zip(rows, cols):
		data[cnt,:,:,0]	= img[r:r+patch_size, c:c+patch_size]
		cnt += 1
	
	return data


def extract_patches_random_multiple(images, patch_size, num_patches):
	
	[height, width]	= images[0].shape
	
	rows	= np.random.randint(height-patch_size+1, size=num_patches)
	cols	= np.random.randint(width -patch_size+1, size=num_patches)
	
	data	= [None]*len(images)
	for i, img in enumerate(images):
		data[i]	= np.empty([num_patches, patch_size, patch_size, 1])
		cnt	= 0
		for r, c in zip(rows, cols):
			data[i][cnt,:,:,0]	= img[r:r+patch_size, c:c+patch_size]
			cnt += 1
	
	return data


def extract_patches(img, patch_size):
	
	[height, width]	= img.shape
	
	num_row	= int(height/float(patch_size))
	num_col	= int(width/float(patch_size))
	
	data	= np.empty([num_row*num_col, patch_size, patch_size, 1])
	cnt	= 0
	for i in range(num_row):
		for j in range(num_col):
			data[cnt,:,:,0]	= img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
			cnt += 1
	
	return data


def get_optimal_weights(M):
	
	if not np.all(np.linalg.eigvals(M) > 0):
		w, v	= np.linalg.eig(M)
		w	= w.clip(min=1e-6)
		M	= np.dot(np.dot(v, np.diag(w)), v.T)
	
	W		= cvx.Variable(shape=(M.shape[0], 1))
	constraints	= [cvx.sum(W)==1, cvx.min(W)>=0]
	obj		= cvx.Minimize(cvx.quad_form(W, M))
	prob		= cvx.Problem(obj, constraints)
	prob.solve(solver=cvx.ECOS)
	
	weights		= [W.value[i,0] for i in range(M.shape[0])]
	return weights


def combine(img_list, mse_list):
	if len(img_list)!=len(mse_list):
		sys.exit(1)
	
	M	= np.zeros((len(img_list), len(img_list)))
	for i in range(len(img_list)):
		for j in range(len(img_list)):
			M[i,j]	= (mse_list[i] + mse_list[j] - cal_mse(img_list[i], img_list[j])) / 2
	weights	= get_optimal_weights(M)
	img	= sum([weights[i]*img_list[i] for i in range(len(img_list))])
	return img, weights


def load_image(filename):
	return imread(filename, flatten=True)/255.0


def save_image(image, filename):
	toimage(image, cmin=0.0, cmax=1.0).save(filename)


def cal_mse(x, y):
	return np.mean(np.square(x-y))


def cal_psnr(x, y):
	return -10*np.log10(cal_mse(x, y))


def cal_SURE(x, yhat, yhat_, sigma_f, b, Eps, N):
	Div	= np.dot(b.flatten(), (yhat_ - yhat).flatten())/Eps
	return cal_mse(x, yhat) - sigma_f**2 + 2*(sigma_f**2)*Div/N


def normalize_img(x, maxv, minv):
	return np.divide(x-minv, maxv-minv)


def augmentation(img, i):
	if 0<=i<=3:
		return np.rot90(img, i)
	elif i==4:
		return np.flip(img, 0)
	elif i==5:
		return np.flip(img, 1)
	else:
		return img

def load_data_test(image, sigma, patch_size, model_0, model_1):
	img_no	= image + sigma/255.0*np.random.normal(size=image.shape)
	img_no4	= np.reshape(img_no, (1, img_no.shape[0], img_no.shape[1], 1))
	img_den	= [model.run(img_no4)[0,:,:,0] for model in model_0]
	mse_est	= [model_1.run(extract_patches(img_no, patch_size), extract_patches(den, patch_size)) for den in img_den]
	img_com, weights	= combine(img_den, mse_est)
	img_com	= np.reshape(img_com, (1, img_com.shape[0], img_com.shape[1], 1))
	return img_no4, img_com


def loading(data, sigma, model_0, model_1):
	img_no	= data + sigma/255.0*np.random.normal(size=data.shape)
	img_den	= [model.run(img_no) for model in model_0]
	mse_est	= [model_1.run(img_no, den) for den in img_den]
	img_com, weights	= combine(img_den, mse_est)
	return img_no, img_com



def load_data(images, patch_size, num_patches, sigmaSet, model_0, model_1):
	data_gt		= np.empty([len(images)*num_patches, patch_size, patch_size, 1])
	data_no		= np.empty([len(images)*num_patches, patch_size, patch_size, 1])
	data_com	= np.empty([len(images)*num_patches, patch_size, patch_size, 1])
	
	idx	= 0
	for img in images:
		data_gt[idx:idx+num_patches]	= extract_patches_random(img, patch_size, num_patches)
		idx	+= num_patches
	np.random.shuffle(data_gt)
	
	for i in range(data_gt.shape[0]):
		data_no[i:i+1],	data_com[i:i+1]	= loading(data_gt[i:i+1], random.choice(sigmaSet), model_0, model_1)
	
	return data_gt, data_no, data_com


def load_data2(images_gt, images_no, images_com, num_patches, patch_size):
	data_gt		= np.empty([len(images_gt) *num_patches, patch_size, patch_size, 1])
	data_no		= np.empty([len(images_no) *num_patches, patch_size, patch_size, 1])
	data_com	= np.empty([len(images_com)*num_patches, patch_size, patch_size, 1])
	
	cnt		= 0
	for i in range(len(images_gt)):
		[height, width]	= images_gt[i].shape
		rows	= np.random.randint(height-patch_size+1, size=num_patches)
		cols	= np.random.randint(width -patch_size+1, size=num_patches)
		
		for r, c in zip(rows, cols):
			data_gt [cnt:cnt+1,:,:,0]	= images_gt [i][r:r+patch_size, c:c+patch_size]
			data_no [cnt:cnt+1,:,:,0]	= images_no [i][r:r+patch_size, c:c+patch_size]
			data_com[cnt:cnt+1,:,:,0]	= images_com[i][r:r+patch_size, c:c+patch_size]
			cnt += 1
	
	indices		= np.arange(data_gt.shape[0])
	np.random.shuffle(indices)
	
	data_gt		= data_gt [indices]
	data_no		= data_no [indices]
	data_com	= data_com[indices]
	return data_gt, data_no, data_com

