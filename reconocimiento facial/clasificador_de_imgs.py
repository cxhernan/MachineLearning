import numpy as np
import pandas as pd
from PIL import Image,ImageOps
from scipy.linalg import svd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import warnings
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
warnings.filterwarnings("ignore")
def get_dados(ruta):
	data_x=pd.read_csv(ruta+"X.csv",header=None)
	data_y=pd.read_csv(ruta+"Y.csv",header=None)
	data_p=pd.read_csv(ruta+"P.csv",header=None)
	X=data_x.iloc[:,:].values
	Y=data_y.iloc[:,:].values
	P=data_p.iloc[:,:].values
	return X,Y,P

def normalizar(X):
	mu=np.mean(X,axis=0)
	sigma=np.std(X,axis=0)
	Xnorm=(X-mu)/sigma
	return Xnorm,mu,sigma 

def convertir_vetor_matriz(vetor,tam_matriz):
	matriz=vetor.reshape(tam_matriz,tam_matriz)
	return matriz

def mostrar_resultados_imgs_test(imgs,prediccion):
	num_imgs=imgs.shape[0]
	tam_img=int(np.sqrt(imgs.shape[1]))
	if(num_imgs>10):
		tam_grid_fil=int(np.floor(np.sqrt(num_imgs)))+1
		tam_grid_col=tam_grid_fil-1
	else:
		tam_grid_fil=1
		tam_grid_col=num_imgs

	fig,axis=plt.subplots(tam_grid_fil,tam_grid_col,figsize=(20,20),sharex=True, sharey=True)
	fig.tight_layout(pad=3.0)
	if(tam_grid_fil==1):
		for i in range(num_imgs):
			vetor_img=imgs[i,:]
			img=convertir_vetor_matriz(vetor_img,tam_img)
			axis[i].imshow(img,cmap="gray")
			axis[i].set_title("Predicción: "+prediccion[i],size=10)
	else:
		print(tam_grid_fil)
		k=0
		for i in range(tam_grid_fil):
			for j in range(tam_grid_col):
				if(k>num_imgs-1):
					break
				vetor_img=imgs[k,:]
				img=convertir_vetor_matriz(vetor_img,tam_img)
				axis[i,j].imshow(img,cmap="gray")
				axis[i,j].set_title("Predicción: "+prediccion[k],size=5)
				#axis[i,j].axis("off")	
				k=k+1
			if(k>num_imgs-1):
				break

def mostrar_dados(X,Y,cores=['red','blue'],labels=['0','1']):
	indices_0,indices_1=(Y==0),(Y==1)
	indices_0,indices_1=indices_0[:,0],indices_1[:,0]
	x0,y0=X[indices_0,0],X[indices_0,1]
	x1,y1=X[indices_1,0],X[indices_1,1]
	plt.scatter(x0,y0,color=cores[0],marker='x',label=labels[0])
	plt.scatter(x1,y1,color=cores[1],label=labels[1])
	plt.title('Representação das imagens no plano')
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.legend()

def dividir_conjunto_dados_aleatoriamente(X,Y,porc=0.33):
	pos_train_test = list(StratifiedShuffleSplit(n_splits=2,test_size=porc).split(X,Y))[0]
	pos_train,pos_test=pos_train_test[0],pos_train_test[1]
	Xtrain,Ytrain=X[pos_train],Y[pos_train]
	Xtest,Ytest=X[pos_test],Y[pos_test]
	return Xtrain,Ytrain,Xtest,Ytest

def get_autovetores_da_matriz_cov(X):
	U,D,VT=svd(X)
	V=VT.T
	return V

"""Para representar uma imagem no plano, precisamos projetar ela sobre 2 vetores ortonormales
para obter uma coordenada de dimensão 2"""
def get_dois_autovetores(V,x,y):
	return V[:,[x,y]]

def projetar_imgs(X,V):
	return X @ V

def treinar_reconhecimento_face(X,Y):
	Xtrain,Ytrain,Xval,Yval=dividir_conjunto_dados_aleatoriamente(X,Y,porc=0.2)
	lista_C=[0.01,0.03,0.1,0.3,1,3,10,30]
	lista_sigma=[0.01,0.03,0.1,0.3,1,3,10,30]
	max_accuracy=0
	for C in lista_C:
		for sigma in lista_sigma:
			gamma=1/(2*sigma)
			classifier=SVC(C=C,kernel="linear",gamma=gamma)
			classifier.fit(X,Y[:,0])
			pred_train=classifier.predict(X)
			pred_val=classifier.predict(Xval)
			accuracy_train=1-np.count_nonzero(pred_train-Y.ravel())/len(pred_train)
			accuracy_val=1-np.count_nonzero(pred_val-Yval.ravel())/len(pred_val)
			if(accuracy_val>max_accuracy):
				max_accuracy=accuracy_val
				best_C=C
				best_sigma=sigma
				best_classifier=classifier
	return best_classifier

def desenhar_fronteira_decisao(X,classifier,core="blue"):
	num_pontos_x_y=1000 
	p_x=np.linspace(X[:,0].min(),X[:,0].max(),num_pontos_x_y) 
	p_y=np.linspace(X[:,1].min(),X[:,1].max(),num_pontos_x_y) 
	g_x,g_y=np.meshgrid(p_x, p_y)
	g_X_reshaped=np.array([g_x.ravel(),g_y.ravel()]).T
	pred=classifier.predict(g_X_reshaped)
	g_pred=pred.reshape(g_x.shape)
	contour=plt.contour(g_x,g_y,g_pred,[1],colors=core)

def reconhecer_pessoas(X,P,classificador):
	Y=classificador.predict(X)
	pessoas_reconhecidas=P[Y,1]
	return pessoas_reconhecidas


#Carregamos os dados, e separamos uma parte para treinamento e outra para teste (de forma estratificada), e normalizamos.
pasta_dados="dados/"
X,Y,P=get_dados(pasta_dados)
Xtrain,Ytrain,Xtest,Ytest=dividir_conjunto_dados_aleatoriamente(X,Y)
Xtrain_norm,mu,sigma=normalizar(Xtrain)
Xtest_norm=(Xtest-mu)/sigma

#Projetamos as imagens no plano e mostramos as nuvens de pontos geradas pelas imagens de treinamento e teste.
V=get_autovetores_da_matriz_cov(Xtrain_norm)
base_projecao=get_dois_autovetores(V,0,1)
Xtrain_norm_plano=projetar_imgs(Xtrain_norm,base_projecao)
Xtest_norm_plano=projetar_imgs(Xtest_norm,base_projecao)
mostrar_dados(Xtrain_norm_plano,Ytrain,cores=['red','blue'],labels=['Train: '+str(P[0,1]),'Train: '+str(P[1,1])])
mostrar_dados(Xtest_norm_plano,Ytest,cores=['magenta','cyan'],labels=['Test: '+str(P[0,1]),'Test: '+str(P[1,1])])

#Usamos SVM 'otimizado' para obter um classificador treinado com o conjunto de treinamento e desenhamos a fronteira de decisão.
classificador=treinar_reconhecimento_face(Xtrain_norm_plano,Ytrain)
desenhar_fronteira_decisao(Xtrain_norm_plano,classificador,core="green")

#Fazemos predições com o conjunto de teste e mostramos as imagens (fotos) com sua respectiva predição.
pessoas_reconhecidas=reconhecer_pessoas(Xtest_norm_plano,P,classificador)
mostrar_resultados_imgs_test(Xtest,pessoas_reconhecidas)

plt.show()