import os
import numpy as np
from decord import VideoReader, cpu
import pandas as pd

import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from src.transformers.data.datasets.transforms import video_transforms, volume_transforms


class VideoDataset(torch.utils.data.Dataset):

    def __init__(self, mode='train', clip_len=8, frame_sample_rate=8, crop_size=224,
                 short_side=256, num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3,
                 video_dir='', label_dir=''):
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side = short_side
        self.num_segment = num_segment
        self.num_crop = num_crop
        self.test_num_segment = test_num_segment
        self.test_num_crop = test_num_crop
        self.video_dir = video_dir
        self.label_dir = label_dir

        if mode == 'train':
            cleaned = pd.read_csv(label_dir, header=None, delimiter=' ')
            self.dataset_samples = list(cleaned.values[:, 0])
            self.label_array = list(cleaned.values[:, 2])

            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(self.short_side, interpolation='bilinear'),
                video_transforms.RandomResize(ratio=(1, 1.25), interpolation='bilinear'),
                video_transforms.RandomHorizontalFlip(),
                video_transforms.RandomCrop(size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN,
                                           std=IMAGENET_DEFAULT_STD)
            ])
        elif mode == 'validation':
            cleaned = pd.read_csv(label_dir, header=None, delimiter=' ')
            self.dataset_samples = list(cleaned.values[:, 0])
            self.label_array = list(cleaned.values[:, 2])

            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(self.short_side, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN,
                                           std=IMAGENET_DEFAULT_STD)
            ])
        elif mode == 'test':
            cleaned = pd.read_csv(label_dir, header=None, delimiter=' ')
            self.dataset_samples = list(cleaned.values[:, 0])
            self.label_array = list(cleaned.values[:, 2])

            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(self.crop_size, interpolation='bilinear'),
                video_transforms.VideoThreeCrop(size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN,
                                           std=IMAGENET_DEFAULT_STD)
            ])
        else:
            print('Invalid mode. We only support train, validation and test.')

    def __getitem__(self, index):
        sample = self.dataset_samples[index]
        if self.mode == 'train':
            buffer = self.loadvideo_decord(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    index = np.random.randint(len(self.dataset_samples))
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample)
        elif self.mode == 'validation':
            buffer = self.loadvideo_decord(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    index = np.random.randint(len(self.dataset_samples))
                    sample = self.dataset_samples[index][ind]
                    buffer = self.loadvideo_decord(sample)
        elif self.mode == 'test':
            buffer = self.loadvideo_test_decord(sample)
            if len(buffer) == 0:
                err_msg = 'Video file %s cannot be found or read, please check your data.' % sample
                raise RuntimeError(err_msg)
        else:
            print('Invalid mode. We only support train, validation and test.')

        buffer = self.data_transform(buffer)
        return buffer, self.label_array[index], sample.split(".")[0]

    def loadvideo_decord(self, sample):
        fname = '{}'.format(self.video_dir) + sample

        if not (os.path.exists(fname)):
            print("Video not found: ", fname)
            return []

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []

        try:
            vr = VideoReader(fname, ctx=cpu(0))
        except:
            print("Video cannot be loaded by decord: ", fname)
            return []

        converted_len = int(self.clip_len * self.frame_sample_rate)
        if len(vr) <= converted_len:
            index = np.linspace(0, len(vr), num=len(vr) // self.frame_sample_rate)
            index = np.concatenate((index, np.ones(self.clip_len - len(vr) // self.frame_sample_rate) * len(vr)))
            index = np.clip(index, 0, len(vr) - 1).astype(np.int64)
        else:
            end_idx = np.random.randint(converted_len, len(vr))
            str_idx = end_idx - converted_len
            index = np.linspace(str_idx, end_idx, num=self.clip_len)
            index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)

        buffer = vr.get_batch(list(index)).asnumpy()
        return buffer

    def loadvideo_test_decord(self, sample):
        fname = '{}'.format(self.video_dir) + sample

        if not (os.path.exists(fname)):
            print("Video not found: ", fname)
            return []

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []

        try:
            vr = VideoReader(fname, ctx=cpu(0))
        except:
            print("Video cannot be loaded by decord: ", fname)
            return []

        converted_len = int(self.clip_len * self.frame_sample_rate)
        num_frames = len(vr)
        if num_frames < converted_len:
            indices = np.zeros((self.test_num_segment,))
        else:
            tick = (num_frames - converted_len + 1) / float(self.test_num_segment)
            indices = np.array([int(tick * x) for x in range(self.test_num_segment)])

        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, converted_len, self.frame_sample_rate)):
                frame_id = offset
                frame_id_list.append(frame_id)
                if offset + self.frame_sample_rate < num_frames:
                    offset += self.frame_sample_rate
        video_data = vr.get_batch(frame_id_list).asnumpy()
        buffer = [video_data[vid, :, :, :] for vid, _ in enumerate(frame_id_list)]
        return buffer

    def __len__(self):
        return len(self.dataset_samples)


def build_dataloader(cfg):
    # build dataset
    train_dataset = VideoDataset(mode='train', clip_len=cfg.CONFIG.DATA.CLIP_LEN,
                                 frame_sample_rate=cfg.CONFIG.DATA.FRAME_RATE,
                                 video_dir=cfg.CONFIG.DATA.TRAIN_DATA_PATH,
                                 label_dir=cfg.CONFIG.DATA.TRAIN_ANNO_PATH)
    if not cfg.CONFIG.TEST.MULTI_VIEW_TEST:
        # for validation during training
        val_dataset = VideoDataset(mode='validation', clip_len=cfg.CONFIG.DATA.CLIP_LEN,
                                   frame_sample_rate=cfg.CONFIG.DATA.FRAME_RATE,
                                   video_dir=cfg.CONFIG.DATA.VAL_DATA_PATH,
                                   label_dir=cfg.CONFIG.DATA.VAL_ANNO_PATH)
    else:
        # for multiview test
        val_dataset = VideoDataset(mode='test', clip_len=cfg.CONFIG.DATA.CLIP_LEN,
                                   frame_sample_rate=cfg.CONFIG.DATA.FRAME_RATE,
                                   video_dir=cfg.CONFIG.DATA.VAL_DATA_PATH,
                                   label_dir=cfg.CONFIG.DATA.VAL_ANNO_PATH)

    # build data sampler
    if cfg.DDP_CONFIG.DISTRIBUTED:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    # build data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.CONFIG.TRAIN.BATCH_SIZE, shuffle=train_sampler is None,
        num_workers=9, sampler=train_sampler, pin_memory=True)

    if not cfg.CONFIG.TEST.MULTI_VIEW_TEST:
        # for validation during training
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.CONFIG.VAL.BATCH_SIZE, shuffle=(val_sampler is None),
            num_workers=9, sampler=val_sampler, pin_memory=True)
    else:
        # for multiview test
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.CONFIG.TEST.BATCH_SIZE, shuffle=(val_sampler is None),
            num_workers=9, sampler=val_sampler, pin_memory=True)

    return train_loader, val_loader, train_sampler, val_sampler, None
