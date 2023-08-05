from utils.utils_wgan import *

batch_size=2

class ode_dataset(Dataset):
    def __init__(self,path):
        data=torch.load(path).cuda()#[10,100,4]
        self._x=data[0:9,:,0:3]
        self._y = data[0:9, :,3]
        self._len=len(data[0:9,:,0:3])

    def __getitem__(self,item):
        return self._x[item],self._y[item]
    def __len__(self):
        return self._len
set_seed(42)
train_data=ode_dataset('../ode_dataset/train_data.pt')
test_dataset=ode_dataset('../ode_dataset/test_data.pt')
print(train_data.__len__())
ode_dataloader=DataLoader(train_data,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
print("batch_size:",batch_size)

