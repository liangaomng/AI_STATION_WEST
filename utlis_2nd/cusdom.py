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
        data = self.data[index].to(dtype=torch.float32)
        label = self.label[index].to(dtype=torch.float32)
        # pre-processing
        return data, label

    def __len__(self):
        return self.length


def read_yaml_file(file_path):
    '''
    :param file_path:  'expr2.yaml'
    :return: data in the yaml like a dcit
    '''
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def return_train_valid_test4loader(abso_path="",
                                   yaml_path='expr2.yaml'):

    '''
    input params: 1.dataset_path
                  2.batch_size
                  3.yaml_path
                  4.seed
    :return: loader of train,valid,test,and t_steps
    '''
    print("load data & read yaml")
    yaml_config = read_yaml_file(yaml_path)
    torch.manual_seed(yaml_config["seed"])
    custom_data=CustomDataset(abso_path)

    yaml_config["all_solus_numbers"]=len(custom_data)
    #split dataset
    train_size = int(yaml_config["train_size_percent"] * len(custom_data))
    valid_size= int(yaml_config["valid_size_percent"]*len(custom_data))
    test_size =  int(yaml_config["test_size_percent"]*len(custom_data))
    train_dataset,valid_dataset,test_dataset = random_split(custom_data, [train_size,valid_size,test_size])

    train_loader=DataLoader(train_dataset,batch_size=yaml_config["batch_size"],
                            shuffle=True,drop_last=True,num_workers=4)
    valid_loader=DataLoader(valid_dataset,batch_size=yaml_config["batch_size"],
                            shuffle=True,drop_last=True,num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=yaml_config["batch_size"],
                             shuffle=True, drop_last=True, num_workers=4)

    t_steps =  train_loader.dataset[1][0].shape[0]
    vesting =  train_loader.dataset[1][0][t_steps-1][6] #unit is "s"

    vari_numbs = yaml_config["vari_number"]
    dt=train_loader.dataset[1][0][1][6]-train_loader.dataset[1][0][0][6] #t1-t0 unit(s)


    sample_rate=torch.reciprocal(dt)
    freq_index = torch.fft.rfftfreq(t_steps, d=dt)
    freq_numbers = torch.tensor(freq_index.shape[0])

    yaml_config["data_description"]=[t_steps,vesting.item(),vari_numbs,sample_rate.item(),freq_numbers.item()]

    return train_loader,\
        valid_loader,\
        test_loader,\
        yaml_config\










