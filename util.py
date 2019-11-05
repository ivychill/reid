
import torch
import os

def save_network(network, save_dir, epoch_label, gpu_ids):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',save_dir,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])

def load_network(network, save_dir, epoch_label):
    save_path = os.path.join('./model',save_dir,'net_%s.pth'%epoch_label)
    network.load_state_dict(torch.load(save_path))
    return network