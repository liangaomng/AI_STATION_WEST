
import pandas as pd
'''
该文档主要是为了读取参数，然后进行测试
读取文件为csv
'''
def Get_test_args()->list:
    '''
    :return: 返回一个list，list中的元素为dic\
    '''
    df=pd.read_csv('../wgan_v0_z_dim/test_argus.csv',header=0)
    #读取有效的列 列名作为dict的key
    # 提取第一行作为字典的键，第二行作为字典的值
    keys=df.columns.values.tolist()
    print(keys)
    #第二行作为字典的值
    values=df.values.tolist()
    #将两个list合并成一个dict
    argu_dict=[dict(zip(keys,values[i])) for i in range(len(values))]
    for i in range(len(argu_dict)):
        for key in argu_dict[i].keys():
            #去掉逗号
            if type(argu_dict[i][key])==str:
                data_values=argu_dict[i][key].split(",")
                data_values=[float(data_values[i]) for i in range(len(data_values))]
                argu_dict[i][key]=data_values
    #如何去掉外面的括号
    return argu_dict
