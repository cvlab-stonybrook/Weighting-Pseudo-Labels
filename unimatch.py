import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
from util.feature_memory import FeatureMemory

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
import setproctitle
#setproctitle.setproctitle("Please email if u need gpu")


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus(cfg)
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
    #print(cfg)
    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path, bbox_path=cfg['bbox_path'], bbox_lbl_path=cfg['bbox_lbl_path'], bbox_score_path=cfg['bbox_score_path'])
    #print(trainset_u[0])
    #print(trainset_u[1])
    #print(trainset_u[0])
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)
    feature_memory = FeatureMemory(num_samples=len(trainset_l), dataset=cfg['dataset'], memory_per_class=256, feature_size=256, n_classes=cfg['nclass'])
    feature_memory_unlab = FeatureMemory(num_samples=len(trainset_u), dataset=cfg['dataset'], memory_per_class=256, feature_size=256, n_classes=cfg['nclass'])
    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    
    if os.path.exists(os.path.join(args.save_path, '165_latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, '165_latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2,BB_all_u_w_s1),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _, BB_all_u_w_mix)) in enumerate(loader):
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()#[8, 3, 321, 321]
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()#[8, 321, 321]
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()
            BB_all_u_w_s1 = BB_all_u_w_s1.cuda()#[8, 21, 321, 321]
            BB_all_u_w_mix = BB_all_u_w_mix.cuda()
            BB_all_u_w_s2 = BB_all_u_w_s1.clone()

            with torch.no_grad():
                model.eval()
                feat_u_w_mix,pred_u_w_mix = model(img_u_w_mix, need_feat=True)#[8, 256, 321, 321], [8, 21, 321, 321]
                pred_u_w_mix = pred_u_w_mix.detach()
                feat_u_w_mix  = feat_u_w_mix .detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            BB_all_u_w_s1[cutmix_box1.unsqueeze(1).expand(BB_all_u_w_s1.shape) == 1] = \
                BB_all_u_w_mix[cutmix_box1.unsqueeze(1).expand(BB_all_u_w_mix.shape) == 1]
            BB_all_u_w_s2[cutmix_box2.unsqueeze(1).expand(BB_all_u_w_s2.shape) == 1] = \
                BB_all_u_w_mix[cutmix_box2.unsqueeze(1).expand(BB_all_u_w_mix.shape) == 1]
            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            feature, feature_fp,preds, preds_fp = model(torch.cat((img_x, img_u_w)), True, need_feat=True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])#[8, 21, 321, 321],[8, 21, 321, 321]
            feature_x, feature_u_w = feature.split([num_lb, num_ulb])#[8, 256, 321, 321]), ([8, 256, 321, 321]
            pred_u_w_fp = preds_fp[num_lb:]
            conf_p_x = pred_x.softmax(dim=1).max(dim=1)[0]#[8, 321, 321]
            mask_p_x = pred_x.argmax(dim=1)
            loss_x = criterion_l(pred_x, mask_x)
            if torch.isnan(loss_x).any():
                print("NAN in loss_x")
                print(pred_x.shape, mask_x.shape)
                print(torch.unique(pred_x), torch.unique(mask_x))
                loss_x = torch.nan_to_num(loss_x)
                #####################################################################################################
            with torch.no_grad():
                # get mask where the labeled predictions are correct and have a confidence higher than 0.95
                mask_prediction_correctly = ((mask_p_x == mask_x).float() * (conf_p_x > 0.95).float()).bool()#[8, 321, 321]   
                #print( mask_prediction_correctly.int().sum())
                # Apply the filter mask to the features and its labels
                feature_x = feature_x.permute(0, 2, 3, 1)
                labels_down_correct = mask_p_x[mask_prediction_correctly]#[27739]
                labeled_features_correct = feature_x[mask_prediction_correctly, ...]#[27739, 256]
                # updated memory bank
                feature_memory.add_features_from_sample_learned(model, labeled_features_correct, labels_down_correct, mask_x.shape[0])
            #loss_x = criterion_l(pred_x, mask_x)            
            feats_u,preds_u = model(torch.cat((img_u_s1, img_u_s2)),need_feat=True)
            pred_u_s1, pred_u_s2= preds_u.chunk(2)#[8, 21, 321, 321] [8, 21, 321, 321]
            feats_u_s1, feats_u_s2 = feats_u.chunk(2)#[8, 256, 321, 321]) ([8, 256, 321, 321]

            #preds_u = model(torch.cat((img_u_s1, img_u_s2)))
            #pred_u_s1, pred_u_s2= preds_u.chunk(2)#[8, 21, 321, 321] [8, 21, 321, 321]
            #feats_u_s1, feats_u_s2 = feats_u.chunk(2)#[8, 256, 321, 321]) ([8, 256, 321, 321]
            with torch.no_grad():
                #feats_u_arg_s1 = torch.argsort(feats_u_s1,dim=1, descending=True)[:,:5,:,:]#[8, 5, 321, 321]
                #feats_u_arg_s1,_ = torch.sort(feats_u_arg_s1,dim=1)
                #feats_u_arg_s1 = feats_u_arg_s1.detach()
                #feats_u_arg_s2 = torch.argsort(feats_u_s2,dim=1, descending=True)[:,:5,:,:]#[8, 5, 321, 321]
                #feats_u_arg_s2,_ = torch.sort(feats_u_arg_s2,dim=1) 
                #feats_u_arg_s2 = feats_u_arg_s2.detach()
                feature_u_w_arg = torch.argsort(feature_u_w,dim=1, descending=True)[:,:cfg['K'],:,:]#[8, 5, 321, 321]
                #feature_u_w_arg,_ = torch.sort(feature_u_w_arg,dim=1)
                feature_u_w_arg = feature_u_w_arg.detach()
                feat_u_w_mix_arg = torch.argsort(feat_u_w_mix,dim=1, descending=True)[:,:cfg['K'],:,:]#[8, 5, 321, 321]
                #feat_u_w_mix_arg,_ = torch.sort(feat_u_w_mix_arg,dim=1)
                feat_u_w_mix_arg = feat_u_w_mix_arg.detach()
                feature_u_w_arg_s1, feature_u_w_arg_s2 = feature_u_w_arg.clone(), feature_u_w_arg.clone()
                feature_u_w_arg_s1[cutmix_box1.unsqueeze(1).expand(feature_u_w_arg_s1.shape) == 1] = \
                feat_u_w_mix_arg[cutmix_box1.unsqueeze(1).expand(feat_u_w_mix_arg.shape) == 1]
                feature_u_w_arg_s2[cutmix_box2.unsqueeze(1).expand(feature_u_w_arg_s2.shape) == 1] = \
                feat_u_w_mix_arg[cutmix_box2.unsqueeze(1).expand(feat_u_w_mix_arg.shape) == 1]

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]
            ###############################################################################################################################################later
            #mask_prediction_correctly_cutmix1 = ((ignore_mask_cutmixed1 != 255).float() * (conf_u_w_cutmixed1 >= cfg['conf_thresh']).float()).bool()#[4, 321, 321]
            #mask_prediction_correctly_cutmix2 = ((ignore_mask_cutmixed2 != 255).float() * (conf_u_w_cutmixed2 >= cfg['conf_thresh']).float()).bool()#[4, 321, 321]
            #print(mask_u_w_cutmixed1.shape, BB_all_u_w_s1.shape )#([4, 321, 321]) ([4, 21, 321, 321])
            #print(torch.unique(BB_all_u_w_s1))#[  0,   1, 255],
            with torch.no_grad():
                change_mask_u_w_cutmixed1 = mask_u_w_cutmixed1.clone()
                change_mask_u_w_cutmixed2 = mask_u_w_cutmixed2.clone()
                for _class_ in range(cfg['nclass']):
                    temp_class_mask1 = torch.zeros(mask_u_w_cutmixed1.shape).cuda()#4, 321, 321
                    temp_class_mask1[mask_u_w_cutmixed1==_class_]=1
                    temp_class_mask1[BB_all_u_w_s1[:,_class_,:,:]!=1]=0
                    change_mask_u_w_cutmixed1[(change_mask_u_w_cutmixed1==_class_) & (temp_class_mask1==0)]=255
                    temp_class_mask2 = torch.zeros(mask_u_w_cutmixed2.shape).cuda()#4, 321, 321
                    temp_class_mask2[mask_u_w_cutmixed2==_class_]=1
                    temp_class_mask2[BB_all_u_w_s2[:,_class_,:,:]!=1]=0
                    change_mask_u_w_cutmixed2[(change_mask_u_w_cutmixed2==_class_) & (temp_class_mask2==0)]=255
                mask_prediction_correctly_cutmix1 = ((change_mask_u_w_cutmixed1 != 255).float() * (ignore_mask_cutmixed1 != 255).float() * (conf_u_w_cutmixed1 >= 0.95).float()).bool()#[4, 321, 321]
                mask_prediction_correctly_cutmix2 = ((change_mask_u_w_cutmixed2 != 255).float()*(ignore_mask_cutmixed2 != 255).float() * (conf_u_w_cutmixed2 >= 0.95).float()).bool()#[4, 321, 321]
                # Apply the filter mask to the features and its labels
                feats_u_s1 = feats_u_s1.permute(0, 2, 3, 1)
                labels_down_correct = mask_u_w_cutmixed1[mask_prediction_correctly_cutmix1]#[3197]
                labeled_features_correct = feats_u_s1[mask_prediction_correctly_cutmix1, ...]#[3197, 256]
                # updated memory bank
                feature_memory_unlab.add_features_from_sample_learned(model, labeled_features_correct, labels_down_correct, mask_u_w_cutmixed1.shape[0])
                feats_u_s2 = feats_u_s2.permute(0, 2, 3, 1)
                labels_down_correct = mask_u_w_cutmixed2[mask_prediction_correctly_cutmix2]#[3197]
                labeled_features_correct = feats_u_s2[mask_prediction_correctly_cutmix2, ...]#[3197, 256]                
                feature_memory_unlab.add_features_from_sample_learned(model, labeled_features_correct, labels_down_correct, mask_u_w_cutmixed2.shape[0])                
            #print(aaa)
            ##############################################################################################################################################
            with torch.no_grad():
                similar_s1=torch.zeros((mask_u_w_cutmixed1.shape[0], cfg['nclass'],mask_u_w_cutmixed1.shape[1], mask_u_w_cutmixed1.shape[2])).cuda().detach()#[8, 21, 321, 321]
                similar_s2=torch.zeros((mask_u_w_cutmixed2.shape[0], cfg['nclass'],mask_u_w_cutmixed2.shape[1], mask_u_w_cutmixed2.shape[2])).cuda().detach()#[8, 21, 321, 321]
                similar_u_w=torch.zeros((mask_u_w.shape[0], cfg['nclass'],mask_u_w.shape[1], mask_u_w.shape[2])).cuda().detach()#[8, 21, 321, 321]
                for _class_ in list(range(cfg['nclass'])):
                    if(_class_ != 255):
                        memory_c=feature_memory.memory[_class_]
                        memory_c_unlab = feature_memory_unlab.memory[_class_]
                        if(memory_c is not None):
                            memory_c = torch.from_numpy(memory_c).cuda().detach()#M*256
                            #memory_c = torch.mean(memory_c, dim=0)#256
                            if(memory_c_unlab is not None):
                                memory_c_unlab = torch.from_numpy(memory_c_unlab).cuda().detach()#M*256
                                memory_c = torch.cat((memory_c,memory_c_unlab),0)
                            memory_c = torch.mean(memory_c, dim=0)#256
                            memory_c = torch.argsort(memory_c,dim=0, descending=True)[:cfg['K']]
                            for _elem_ in memory_c:
                                similar_s1[:,_class_,:,:]+=(feature_u_w_arg_s1==_elem_).int().sum(1)#[8, 321, 321]
                                similar_s2[:,_class_,:,:]+=(feature_u_w_arg_s2==_elem_).int().sum(1)#[8, 321, 321]
                                similar_u_w[:,_class_,:,:]+=(feature_u_w_arg==_elem_).int().sum(1)
                            similar_s1[:,_class_,:,:][mask_u_w_cutmixed1!=_class_]=0
                            similar_s2[:,_class_,:,:][mask_u_w_cutmixed2!=_class_]=0
                            similar_u_w[:,_class_,:,:][mask_u_w!=_class_]=0
                similar_s1=torch.sum(similar_s1, dim=1)#[8, 321, 321]
                similar_s2=torch.sum(similar_s2, dim=1)
                similar_u_w=torch.sum(similar_u_w, dim=1)
                similar_s1=((similar_s1+0.00000001)/(cfg['K']+0.00000001))+1
                similar_s2=((similar_s2+0.00000001)/(cfg['K']+0.00000001))+1
                similar_u_w=((similar_u_w+0.00000001)/(cfg['K']+0.00000001))+1

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)*similar_s1
            ###check
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (similar_s1*(ignore_mask_cutmixed1 != 255)).sum().item()



            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)*similar_s2
            ######check
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (similar_s2*(ignore_mask_cutmixed2 != 255)).sum().item()

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)*similar_u_w
            #######check
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255))
            loss_u_w_fp = loss_u_w_fp.sum() / (similar_u_w*(ignore_mask != 255)).sum().item()

            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5) / 2.0

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())

            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                if(epoch >= 2):
                    writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                    writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
                    writer.add_scalar('train/mask_ratio', mask_ratio, iters)
            
            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg,
                                            total_loss_w_fp.avg, total_mask_ratio.avg))

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, str(epoch)+'_latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
Your/Pascal/Path