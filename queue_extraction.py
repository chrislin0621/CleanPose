import torch
import pickle as cPickle
from provider.housecat6d_dataset import HouseCat6DTrainingDataset
from provider.nocs_dataset import TrainingDataset
from load_ulip_model import load_ULIP_PN


def create_dataloaders(cfg):
    data_loader = {}
    
    if cfg.dataset_name == "housecat6d":
        real_dataset = HouseCat6DTrainingDataset(
            cfg.image_size, cfg.sample_num, cfg.dataset_dir, cfg.seq_length, cfg.img_length)
        
        real_dataloader = torch.utils.data.DataLoader(real_dataset,
            batch_size=cfg.batchsize,
            num_workers=int(cfg.num_workers),
            shuffle=cfg.shuffle,
            sampler=None,
            drop_last=cfg.drop_last,
            pin_memory=cfg.pin_memory)
        
        data_loader['real'] = real_dataloader
    
    elif cfg.dataset_name == "camera_real":
        syn_dataset = TrainingDataset(
            cfg.image_size, cfg.sample_num, cfg.dataset_dir, 'syn',
            num_img_per_epoch=cfg.num_mini_batch_per_epoch*cfg.syn_bs, threshold=cfg.outlier_th)
            
        syn_dataloader = torch.utils.data.DataLoader(syn_dataset,
            batch_size=cfg.syn_bs,
            num_workers=cfg.syn_num_workers,
            shuffle=cfg.shuffle,
            sampler=None,
            drop_last=cfg.drop_last,
            pin_memory=cfg.pin_memory)
            
        real_dataset = TrainingDataset(
            cfg.image_size, cfg.sample_num, cfg.dataset_dir, 'real_withLabel',
            num_img_per_epoch=cfg.num_mini_batch_per_epoch*cfg.real_bs, threshold=cfg.outlier_th)
            
        real_dataloader = torch.utils.data.DataLoader(real_dataset,
            batch_size=cfg.real_bs,
            num_workers=cfg.real_num_workers,
            shuffle=cfg.shuffle,
            sampler=None,
            drop_last=cfg.drop_last,
            pin_memory=cfg.pin_memory)

        data_loader['syn'] = syn_dataloader
        data_loader['real'] = real_dataloader
    
    else:
        raise NotImplementedError
    
    return data_loader

if __name__ == "__main__":
    
    # # ############# pre extra tensor list ####################
    pre_pts_encoder_pointnet = load_ULIP_PN()
    pre_pts_encoder_pointnet.eval()
    import gorilla
    cfg = gorilla.Config.fromfile('config/REAL/camera_real.yaml')
    print(cfg.train_dataset)
    dataloaders = create_dataloaders(cfg.train_dataset)
    for k in dataloaders.keys():
        dataloaders[k].dataset.reset()
    data_iter = zip(dataloaders["syn"], dataloaders["real"])
    
    
    num_categories = 6 # 10 for housecat6d
    num_features = 80 # 100 for housecat6d
    feature_dim = 256
    tensor_list = [[] for _ in range(num_categories)]
    
    full = False
    for train_data in data_iter:
        syn_data, real_data = train_data
        pts = torch.cat((syn_data['pts'],real_data['pts']), dim=0)
        cls = torch.cat((syn_data['category_label'],real_data['category_label']), dim=0).squeeze(1)
        _,pts_feat = pre_pts_encoder_pointnet.encode_pc(pts.cuda())
        
        for i in range(pts.size(0)):
            cls_idx = cls[i].item()
            if len(tensor_list[cls_idx]) < num_features:
                tensor_list[cls_idx].append(pts_feat[i].unsqueeze(0).cpu())
                
            if all(len(sub_list) >= num_features for sub_list in tensor_list):
                full = True
                break
        if full:
            break

    with open('/PATH/TO/tensor_list.pkl', 'wb') as f:
        cPickle.dump(tensor_list, f)
