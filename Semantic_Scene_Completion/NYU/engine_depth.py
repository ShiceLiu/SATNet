import os
import shutil
import time

import cv2
import numpy as np
import math

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.nn import Parameter
import visdom

import IOU

class Engine(object):
    def __init__(self, state={}):

        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('batch_size') is None:
            self.state['batch_size'] = 1

        if self._state('workers') is None:
            self.state['workers'] = 2

        if self._state('multi_gpu') is None:
            self.state['multi_gpu'] = True

        if self._state('device_ids') is None:
            self.state['device_ids'] = [0]

        if self._state('evaluate') is None:
            self.state['evaluate'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 80

        if self._state('image_visdom_iters') is None:
            self.state['image_visdom_iters'] = self.state['print_freq']

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []

        if self._state('save_iter') is None:
            self.state['save_iter'] = 300

        # meters
        self.state['meter_loss_total'] = 0.0
        self.state['meter_loss_num'] = 0
        # time measure
        self.state['batch_time_total'] = 0.0
        self.state['batch_time_num'] = 0
        self.state['data_time_total'] = 0.0
        self.state['data_time_num'] = 0
        # display parameters
        if self._state('print_freq') is None:
            self.state['print_freq'] = 1

        self.vis = visdom.Visdom()
        self.linewin = self.vis.line(X=np.array([0,0],dtype=int), Y=np.array([2,2],dtype=float))
        self.visindex = 2
        self.visbatchloss = [0.5]
        self.visloss = [0.5]

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        self.state['meter_loss_total'] = 0.0
        self.state['meter_loss_num'] = 0
        self.state['batch_time_total'] = 0.0
        self.state['batch_time_num'] = 0
        self.state['data_time_total'] = 0.0
        self.state['data_time_num'] = 0

        self.state['accuracy_total'] = np.zeros((3, 11), dtype = np.float32)

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        loss = self.state['meter_loss_total'] / self.state['meter_loss_num']
        accuracy = self.state['accuracy_total'][0] / (np.sum(self.state['accuracy_total'], 0) + 0.00001)
        accuracy_mean = np.mean(accuracy)
        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}\t'
                      'Accuracy {accuracy:.4f}'.format(self.state['epoch'], loss=loss, accuracy=accuracy_mean))
            else:
                print('Test: \t Loss {loss:.4f} \t Accuracy {accuracy:.4f}'.format(loss=loss, accuracy=accuracy_mean))
            print 'Obj_class: ' + np.array_str(accuracy)
        return loss, accuracy_mean

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        # record loss
        self.state['loss_batch'] = self.state['loss'].data[0]
        self.state['meter_loss_total'] = self.state['meter_loss_total'] + self.state['loss_batch']
        self.state['meter_loss_num'] = self.state['meter_loss_num'] + 1

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss_total'] / self.state['meter_loss_num']
            batch_time = self.state['batch_time_total'] / self.state['batch_time_num']
            data_time = self.state['data_time_total'] / self.state['data_time_num']
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})\t'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
                
                if self.visindex == 2:
                    self.visloss[0] = loss
                    self.visbatchloss[0] = self.state['loss_batch']
                self.visbatchloss.append(self.state['loss_batch'])
                self.visloss.append(loss)
                self.visindex = self.visindex + 1
                self.vis.line(X=np.column_stack([np.arange(1,self.visindex),np.arange(1,self.visindex)]),
                    Y=np.column_stack([np.asarray(self.visbatchloss),np.asarray(self.visloss)]), win=self.linewin)
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):

        input_var = torch.autograd.Variable(self.state['input'], requires_grad=True)
        target_var = torch.autograd.Variable(self.state['target'], requires_grad=False)
        target_weight_var = torch.autograd.Variable(self.state['target_weight'], requires_grad=False)
        depth_mapping_3d_var = torch.autograd.Variable(self.state['depth_mapping_3d'], requires_grad=False)

        if training:
            self.state['output'] = model(input_var, depth_mapping_3d_var)
            self.selectindex = torch.nonzero(target_weight_var.view(-1)).view(-1)
            self.filterLabel = torch.index_select(target_var.view(-1), 0, self.selectindex)
            self.filterOutput = torch.index_select(self.state['output'].permute(
                0,2,3,4,1).contiguous().view(-1,12), 0, self.selectindex)
            self.state['loss'] = criterion(self.filterOutput, self.filterLabel)
            
            self.state['accuracy_total'] += IOU.computeIOU(self.state['output'].data, self.state['target'], 12)

            optimizer.zero_grad()
            self.state['loss'].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                self.state['output'] = model(input_var, depth_mapping_3d_var)
                self.selectindex = torch.nonzero(target_weight_var.view(-1)).view(-1)
                self.filterLabel = torch.index_select(target_var.view(-1), 0, self.selectindex)
                self.filterOutput = torch.index_select(self.state['output'].permute(
                    0,2,3,4,1).contiguous().view(-1,12), 0, self.selectindex)
                self.state['loss'] = criterion(self.filterOutput, self.filterLabel)

                self.state['accuracy_total'] += IOU.computeIOU(self.state['output'].data, self.state['target'], 12)

    def init_learning(self, model, criterion):

        self.state['best_score'] = 0

    def learning(self, model, criterion, train_dataset, val_dataset, optimizer=None):

        self.init_learning(model, criterion)

        # data loading code
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.state['batch_size'], shuffle=True,
                                                   num_workers=self.state['workers'])

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.state['batch_size'], shuffle=False,
                                                 num_workers=self.state['workers'])

        # optionally resume from a checkpoint
        if self._state('resume') is not None:
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                checkpoint = torch.load(self.state['resume'])
                self.state['start_epoch'] = checkpoint['epoch']
                self.state['best_score'] = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'], False)
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.state['evaluate'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))


        if self.state['use_gpu']:
            train_loader.pin_memory = True
            val_loader.pin_memory = True
            cudnn.benchmark = True

            if self.state['multi_gpu']:
                model = torch.nn.DataParallel(model, device_ids=self.state['device_ids']).cuda()
            else:
                model = torch.nn.DataParallel(model).cuda(0)

            criterion = criterion.cuda()

        if self.state['evaluate']:
            self.validate(val_loader, model, criterion)
            return

        # TODO define optimizer

        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            self.adjust_learning_rate(optimizer)

            # train for one epoch
            self.train(train_loader, model, criterion, optimizer, epoch)

            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self._state('arch'),
                'state_dict': model.module.state_dict() if self.state['use_gpu'] else model.state_dict(),
                'best_score': self.state['best_score'],
            }, False)

            # evaluate on validation set
            loss1, prec1 = self.validate(val_loader, model, criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > self.state['best_score']
            self.state['best_score'] = max(prec1, self.state['best_score'])
            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self._state('arch'),
                'state_dict': model.module.state_dict() if self.state['use_gpu'] else model.state_dict(),
                'best_score': self.state['best_score'],
            }, is_best)

            print(' *** best={best:.3f}'.format(best=self.state['best_score']))

    def train(self, data_loader, model, criterion, optimizer, epoch):

        # switch to train mode
        model.train()

        self.on_start_epoch(True, model, criterion, data_loader, optimizer)

        end = time.time()
        for i, (color, label, label_weight, depth_mapping_3d) in enumerate(data_loader):
            
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time_total'] = self.state['data_time_total'] + self.state['data_time_batch']
            self.state['data_time_num'] = self.state['data_time_num'] + 1

            self.state['input'] = color
            self.state['target'] = label
            self.state['target_weight'] = label_weight
            self.state['depth_mapping_3d'] = depth_mapping_3d

            self.on_start_batch(True, model, criterion, data_loader, optimizer)

            if self.state['use_gpu']:
                self.state['input'] = self.state['input'].cuda(async=True)
                self.state['target'] = self.state['target'].cuda(async=True)
                self.state['target_weight'] = self.state['target_weight'].cuda(async=True)
                self.state['depth_mapping_3d'] = self.state['depth_mapping_3d'].cuda(async=True)

            self.on_forward(True, model, criterion, data_loader, optimizer)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time_total'] = self.state['batch_time_total'] + self.state['batch_time_current']
            self.state['batch_time_num'] = self.state['batch_time_num'] + 1
            end = time.time()
            # measure accuracy
            self.on_end_batch(True, model, criterion, data_loader, optimizer)

            if self.state['save_iter'] != 0 and i != 0 and i % self.state['save_iter'] == 0:
                print 'save checkpoint once!'
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': self._state('arch'),
                    'state_dict': model.module.state_dict() if self.state['use_gpu'] else model.state_dict(),
                    'best_score': self.state['best_score'],
                }, False)

        self.on_end_epoch(True, model, criterion, data_loader, optimizer)

    def validate(self, data_loader, model, criterion):

        # switch to evaluate mode
        model.eval()

        self.on_start_epoch(False, model, criterion, data_loader)

        end = time.time()
        for i, (color, label, label_weight, depth_mapping_3d) in enumerate(data_loader):

            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time_total'] = self.state['data_time_total'] + self.state['data_time_batch']
            self.state['data_time_num'] = self.state['data_time_num'] + 1

            self.state['input'] = color
            self.state['target'] = label
            self.state['target_weight'] = label_weight
            self.state['depth_mapping_3d'] = depth_mapping_3d

            self.on_start_batch(False, model, criterion, data_loader)

            if self.state['use_gpu']:
                self.state['input'] = self.state['input'].cuda(async=True)
                self.state['target'] = self.state['target'].cuda(async=True)
                self.state['target_weight'] = self.state['target_weight'].cuda(async=True)
                self.state['depth_mapping_3d'] = self.state['depth_mapping_3d'].cuda(async=True)

            self.on_forward(False, model, criterion, data_loader)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time_total'] = self.state['batch_time_total'] + self.state['batch_time_current']
            self.state['batch_time_num'] = self.state['batch_time_num'] + 1
            end = time.time()
            # measure accuracy
            self.on_end_batch(False, model, criterion, data_loader)

        loss, score = self.on_end_epoch(False, model, criterion, data_loader)

        return loss, score

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if self._state('save_model_path') is not None:
            filename_ = filename
            filename = os.path.join(self.state['save_model_path'], filename_)
            if not os.path.exists(self.state['save_model_path']):
                os.makedirs(self.state['save_model_path'])
        print('save model {filename}'.format(filename=filename))
        torch.save(state, filename)

        # my add
        filename_my = os.path.join(self.state['save_model_path'], 'checkpoint_{}.pth.tar'.format(self.state['epoch']+1))
        torch.save(state, filename_my)
        # my add

        if is_best:
            filename_best = 'model_best.pth.tar'
            if self._state('save_model_path') is not None:
                filename_best = os.path.join(self.state['save_model_path'], filename_best)
            shutil.copyfile(filename, filename_best)
            if self._state('save_model_path') is not None:
                if self._state('filename_previous_best') is not None:
                    os.remove(self._state('filename_previous_best'))
                filename_best = os.path.join(self.state['save_model_path'], 'model_best_{score:.4f}.pth.tar'.format(score=state['best_score']))
                shutil.copyfile(filename, filename_best)
                self.state['filename_previous_best'] = filename_best

    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        # lr = args.lr * (0.1 ** (epoch // 30))
        if self.state['epoch'] is not 0 and self.state['epoch'] in self.state['epoch_step']:
            print('update learning rate')
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
                print(param_group['lr'])