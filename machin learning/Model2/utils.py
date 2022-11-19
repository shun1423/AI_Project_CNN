import os
import torch
import matplotlib.pyplot as plt


def save(ckpt_dir, net):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict()},  # state_dict: 모델의 파라미터들, 옵티마이저의 상태 등이 들어있음
               "./%s/model_epoch.pth" % ckpt_dir)            # path


def load(ckpt_dir, net):
    ckpt_lst = os.listdir(ckpt_dir)
    # ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location=torch.device('cpu'))

    net.load_state_dict(dict_model['net'])

    return net

def visual(Q, train_error, test_error, train_acc, test_acc):
    x = [i for i in range(len(train_error) + 1)]
    y_loss_train = [0] + train_error
    y_acc_train = [0] + train_acc

    y_loss_test = [0] + test_error
    y_acc_test = [0] + test_acc


    plt.figure(figsize=(14, 10))

    plt.subplot(2,1,1)
    plt.plot(x, y_loss_train, label='train')
    plt.plot(x, y_loss_test, label='test')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Loss Comparison Between Training and Testing Datasets')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(x, y_acc_train, label='train')
    plt.plot(x, y_acc_test, label='test')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Acuracy Comparison Between Training and Testing Datasets')
    plt.legend()

    plt.savefig(Q + '_Loss.png')


