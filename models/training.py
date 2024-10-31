"""
Train\Test helper, based on awesome previous work by https://github.com/amirmk89/gepc
"""

import os
import time
import shutil
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision

def adjust_lr(optimizer, epoch, lr=None, lr_decay=None, scheduler=None):
    if scheduler is not None:
        scheduler.step()
        new_lr = scheduler.get_lr()[0]
    elif (lr is not None) and (lr_decay is not None):
        new_lr = lr * (lr_decay ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        raise ValueError('Missing parameters for LR adjustment')
    return new_lr


def compute_loss(nll, reduction="mean", mean=0):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "logsumexp":
        losses = {"nll": torch.logsumexp(nll, dim=0)}
    elif reduction == "exp":
        losses = {"nll": torch.exp(torch.mean(nll) - mean)}
    elif reduction == "none":
        losses = {"nll": nll}

    losses["total_loss"] = losses["nll"]

    return losses


class Trainer:
    def __init__(self, args, model, train_loader, test_loader,
                 optimizer_f=None, scheduler_f=None):
        self.model = model
        self.args = args
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.args.image_save_interval = 5
        # Loss, Optimizer and Scheduler
        if optimizer_f is None:
            self.optimizer = self.get_optimizer()
        else:
            self.optimizer = optimizer_f(self.model.parameters())
        if scheduler_f is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler_f(self.optimizer)

    def get_optimizer(self):
        if self.args.optimizer == 'adam':
            if self.args.lr:
                return optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return optim.Adam(self.model.parameters())
        elif self.args.optimizer == 'adamx':
            if self.args.lr:
                return optim.Adamax(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return optim.Adamax(self.model.parameters())
        return optim.SGD(self.model.parameters(), lr=self.args.lr)

    def adjust_lr(self, epoch):
        return adjust_lr(self.optimizer, epoch, self.args.model_lr, self.args.model_lr_decay, self.scheduler)

    def save_checkpoint(self, epoch, is_best=False, filename=None):
        """
        state: {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}
        """
        state = self.gen_checkpoint_state(epoch)
        if filename is None:
            filename = 'checkpoint.pth.tar'

        state['args'] = self.args

        path_join = os.path.join(self.args.ckpt_dir, filename)
        torch.save(state, path_join)
        if is_best:
            shutil.copy(path_join, os.path.join(self.args.ckpt_dir, 'checkpoint_best.pth.tar'))

    def load_checkpoint(self, filename):
        filename = filename
        try:
            checkpoint = torch.load(filename)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.model.set_actnorm_init()
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(filename, checkpoint['epoch']))
        except FileNotFoundError:
            print("No checkpoint exists from '{}'. Skipping...\n".format(self.args.ckpt_dir))
    def _visualize_normalized_pose_tensorboard(self, pose_data, step, writer):
        # 포즈 데이터의 크기는 [N, T, V, F] 형식입니다.
        # N은 배치 크기, T는 시간 스텝, V는 포즈 지점의 수, F는 포즈 지점의 특징 수입니다.
        # 여기서는 T를 이용하여 각 시간 스텝에서의 포즈를 시각화합니다.
        batch_size, time_steps, num_vertices, num_features = pose_data.shape

        # 시간 스텝을 하나씩 반복하며 포즈를 시각화합니다.
        for t in range(time_steps):
            # 해당 시간 스텝의 포즈 데이터를 가져옵니다.
            frame_pose = pose_data[:, t, :, :]  # shape: [N, V, F]
            # 호스트 메모리로 이동시키기
            frame_pose = frame_pose.cpu().numpy()
            # 각 샘플(배치)별로 포즈를 시각화합니다.
            for i in range(batch_size):
                plt.figure()
                plt.title(f'Normalized Pose (Step: {step}, Batch: {i}, Time Step: {t})')
                plt.xlim(-1, 1)  # 이미지 크기에 따라 조절합니다.
                plt.ylim(-1, 1)  # 이미지 크기에 따라 조절합니다.
                plt.scatter(frame_pose[i, :, 0], frame_pose[i, :, 1], c='r')  # x, y 좌표를 사용하여 키포인트를 시각화합니다.

                # 이미지를 파일로 저장 (옵션)
                # save_path = f"pose_image_{step}_{i}_{t}.png"
                # plt.savefig(save_path)

                # 이미지를 텐서보드에 추가합니다.
                # 이미지를 파일로 저장하고, 해당 파일을 읽어서 텐서보드에 추가하는 방식도 가능합니다.
                # writer.add_image(f'Pose Image/Step_{step}/Batch_{i}/TimeStep_{t}', plt.imread(save_path), step)
                writer.add_figure(f'Pose Image/Step_{step}/Batch_{i}/TimeStep_{t}', plt.gcf(), step)

                # 이미지를 닫습니다.
                plt.close()
    def _visualize_normalized_pose_tensorboard(self, pose_data, step, log_writer):
        # 원래 이미지의 크기에 맞게 키포인트 좌표를 배치합니다.
        # pose_data의 크기는 [Frames, Keypoints, Features]입니다.
        # 예를 들어, pose_data의 shape가 (12, 18, 3)이라면, 12프레임에 걸쳐 18개의 키포인트가 각각 (x, y, score) 형태로 있습니다.
        # 이를 (x, y) 형태로 변경하고, 원래 이미지의 크기에 맞게 스케일링합니다.
        print(pose_data.shape)
        normalized_pose = pose_data[:, :, :2]  # (x, y) 좌표만 선택합니다.
        normalized_pose = (normalized_pose + 1) / 2  # [-1, 1] 범위에서 [0, 1] 범위로 변환합니다.

        # 호스트 메모리로 이동시키기
        normalized_pose = normalized_pose.cpu().numpy()

        # 각 프레임별로 키포인트를 시각화하고 이미지를 저장합니다.
        for i, frame_pose in enumerate(normalized_pose):
            plt.figure()
            plt.title('Normalized Pose')
            plt.xlim(0, 1)  # 이미지 크기에 따라 조절합니다.
            plt.ylim(0, 1)  # 이미지 크기에 따라 조절합니다.
            plt.scatter(frame_pose[:, 0], frame_pose[:, 1], c='r')  # x, y 좌표를 사용하여 키포인트를 시각화합니다.

            # 이미지를 파일로 저장
            save_path = f"pose_image_{step}_{i}.png"
            plt.savefig(save_path)
            plt.close()

            # 텐서보드에 이미지를 추가
            image = torchvision.io.read_image(save_path)
            log_writer.add_image(f'Pose Image/Step_{step}/Frame_{i}', image, step)
        
    def train(self, log_writer=None, clip=100):
        time_str = time.strftime("%b%d_%H%M_")
        checkpoint_filename = time_str + '_checkpoint.pth.tar'
        start_epoch = 0
        num_epochs = self.args.epochs
        self.model.train()
        self.model = self.model.to(self.args.device)
        key_break = False
        for epoch in range(start_epoch, num_epochs):
            if key_break:
                break
            print("Starting Epoch {} / {}".format(epoch + 1, num_epochs))
            pbar = tqdm(self.train_loader)
            for itern, data_arr in enumerate(pbar):
                try:
                    data = [data.to(self.args.device, non_blocking=True) for data in data_arr]
                    score = data[-2].amin(dim=-1)
                    label = data[-1]
                    if self.args.model_confidence:
                        samp = data[0]
                    else:
                        samp = data[0][:, :2]
                    z, nll = self.model(samp.float(), label=label, score=score)
                    if nll is None:
                        continue
                    if self.args.model_confidence:
                        nll = nll * score
                    losses = compute_loss(nll, reduction="mean")["total_loss"]
                    losses.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    pbar.set_description("Loss: {}".format(losses.item()))
                    log_writer.add_scalar('NLL Loss', losses.item(), epoch * len(self.train_loader) + itern)
                    
                    # 모델 구조 시각화
                    if itern == 0:
                        # self.visualize_normalized_pose_tensorboard(samp.float(), itern, log_writer)
                        log_writer.add_graph(self.model, samp.float())
                    
                    # # 배치에 포함된 이미지와 모델의 예측 결과 비교하여 이미지 생성 및 저장
                    # if itern % self.args.image_save_interval == 0:
                    #     self.save_images(samp.float(), label, epoch + 1, z, itern)

                except KeyboardInterrupt:
                    print('Keyboard Interrupted. Save results? [yes/no]')
                    choice = input().lower()
                    if choice == "yes":
                        key_break = True
                        break
                    else:
                        exit(1)

            self.save_checkpoint(epoch, filename=checkpoint_filename)
            new_lr = self.adjust_lr(epoch)
            print('Checkpoint Saved. New LR: {0:.3e}'.format(new_lr))

    def test(self):
        self.model.eval()
        self.model.to(self.args.device)
        pbar = tqdm(self.test_loader)
        probs = torch.empty(0).to(self.args.device)
        print("Starting Test Eval")
        for itern, data_arr in enumerate(pbar):
            data = [data.to(self.args.device, non_blocking=True) for data in data_arr]
            score = data[-2].amin(dim=-1)
            if self.args.model_confidence:
                samp = data[0]
            else:
                samp = data[0][:, :2]
            with torch.no_grad():
                z, nll = self.model(samp.float(), label=torch.ones(data[0].shape[0]), score=score)
            if self.args.model_confidence:
                nll = nll * score
            probs = torch.cat((probs, -1 * nll), dim=0)
        prob_mat_np = probs.cpu().detach().numpy().squeeze().copy(order='C')
        return prob_mat_np

    def gen_checkpoint_state(self, epoch):
        checkpoint_state = {'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(), }
        return checkpoint_state
