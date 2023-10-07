import torch

from torch.utils.data import Dataset,DataLoader,random_split

class CustomDataset(Dataset):
    def __init__(self, file_path):
        # readt .pt
        data = torch.load(file_path)
        self.data = data['data']
        self.label = data['label_csv']
        self.length = len(self.data)

    def __getitem__(self, index):
        # get data&label
        # label is a string of csv number
        data = self.data[index]
        label = self.label[index]
        # pre-processing
        return data, label

    def __len__(self):
        return self.length

dataset_path="/liangaoming/conda_lam/neural_find_sol/wgan_2nd/complex_center_dataset"


torch.manual_seed(42)
custom_data=CustomDataset(dataset_path + '/combined_data.pt')
train_size = int(0.8 * len(custom_data))
valid_size= int(0.1*len(custom_data))
test_size =  int(0.1*len(custom_data))
train_dataset,valid_dataset,test_dataset = random_split(custom_data, [train_size,valid_size,test_size])

train_loader=DataLoader(train_dataset,batch_size=256,shuffle=True,drop_last=True,num_workers=4)
valid_loader=DataLoader(valid_dataset,batch_size=256,shuffle=True,drop_last=True,num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, drop_last=True, num_workers=4)
batch__size=256

print(train_loader.dataset[1][0].shape[0])




