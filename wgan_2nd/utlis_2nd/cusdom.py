import torch
import yaml
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


def read_yaml_file(file_path):
    '''
    :param file_path:  'expr.yaml'
    :return: data in the yaml like a dcit
    '''
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data



def return_train_valid_test4loader(path="complex_center_dataset",
                                   batch_size=256,
                                   yaml_path='expr.yaml',
                                   seed=42):

    '''
    input:  1.dataset_path "wgan_2nd/complex_center_dataset"
            2.batch_size
            3.yaml_path
            4.seed

    :return: loader of train,valid,test,and t_steps
    '''
    #read_yaml_file(yaml_path)
    yaml_data=dataset_path=path
    torch.manual_seed(seed)

    custom_data=CustomDataset(dataset_path + '/combined_data.pt')
    train_size = int(0.8 * len(custom_data))
    valid_size= int(0.1*len(custom_data))
    test_size =  int(0.1*len(custom_data))
    train_dataset,valid_dataset,test_dataset = random_split(custom_data, [train_size,valid_size,test_size])

    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=4)
    valid_loader=DataLoader(valid_dataset,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    print(train_loader.dataset[1][0].shape[0])

    return train_loader,\
        valid_loader,\
        test_loader,\
        train_loader.dataset[1][0].shape[0],\
        yaml_data

return_train_valid_test4loader()




