import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

def compute_gallery_score(model_withoutFC,FE_size, test_dataloader,device):
  '''
  this function is used to compute an array 
  we will extract the feature for each images of gallery and store in matrix
  - model_withoutFC : model without FC layers
  - FE_size : number of features of last layer of the model 
  - test_dataloader : it is your gallery dataset /!\ batch size should be 1
  '''

  gallery_load = iter(test_dataloader)
  size_dataloader = len(test_dataloader) #number of samples in gallery
  step_size = round(size_dataloader/100) #used to update progress bar
  
  #if model provided is already only feature extract stage not needed this line
  #model_withoutFC = torch.nn.Sequential(*(list(model.children())[:-1]))

  # generating the dummy matrix who will store
  # the matrix will be (number of feature of last layer x size of datalaoder)
  gallery_matrix = torch.zeros(FE_size,size_dataloader)
  print("### Computing gallery matrix ###")
  for i in range (0, size_dataloader):
    image,_ = next(gallery_load)
    image = image.to(device)
    result = model_withoutFC(image)
    result = result.view(-1,1)
    gallery_matrix[:,i]=result.data.squeeze()
    del result
    if (i % step_size == 0): # progress bar
      print("\r",round(i/size_dataloader*100)," %complete",end='',flush=True)
  print('')
  print("### finished ###")
  return gallery_matrix



def compare_query(model_withoutFC,query,gallery_matrix,FE_size,device,ranking=True):
  '''
  this function is used to search he closest value between query and gallery matrix
  it will give back the indexes in the gallery
  - model: import model that will be used
  - query : image to find
  - gallery_matrix : 
  - ranking : if True, return top 10 ranking, howerver return top 1 (default is True)
  '''
  inf = 99999999
  #model_withoutFC = torch.nn.Sequential(*(list(model.children())[:-1]))
  query_temp = query.to(device)
  gallery_matrix = gallery_matrix.to(device)
  result_query = model_withoutFC(query_temp)
  #result_query = result_query.view(-1,1)
  #result_query = result.resize(1,number_feature)
  
  result_matrix = torch.zeros(gallery_matrix.shape[1],1)

  for i in range(gallery_matrix.shape[1]):
    result_matrix[i]=torch.dist(gallery_matrix[:,i],result_query)

  #print(result_matrix)
  if ranking == False:
    index = torch.argmin(result_matrix)
    return index
  else :
    index = torch.zeros(10,1)
    for j in range(10):
      index_temp = torch.argmin(result_matrix)
      index[j]=index_temp
      result_matrix[index_temp]=inf
    return index

def index_to_gallery_label(index,test_dataset,print=False):
  '''
  this function is used to search he closest value between query and gallery matrix
  it will give back the indexes in the gallery
  - index: value of index given by function compare_query
  - test_dataloader : it is your gallery dataset /!\ batch size should be 1
  '''
  
  plt.figure(figsize=(15,60))
  plt.axis("off")
  if index.shape!=[]:
    result_label = torch.zeros(index.shape[0],1)
    for i in range(index.shape[0]):
      result_label[i]=test_dataset[int(index[i])][1]
      plt.subplot(1,index.shape[0], i+1)
      plt.axis("off")
      plt.title('result :'+str(i+1)) 
      plt.imshow(np.transpose(vutils.make_grid(test_dataset[int(index[i])][0], normalize=True),(1,2,0)))
    return result_label
  else :
    return test_dataset[int(index)][1]

  