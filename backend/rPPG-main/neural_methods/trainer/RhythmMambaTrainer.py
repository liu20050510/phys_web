"""Trainer for RhythmMamba."""
import json
import os
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
import random
from tqdm import tqdm
import config
from evaluation.post_process import calculate_hr
from evaluation.metrics import calculate_metrics
from neural_methods.model.RhythmMamba import  RhythmMamba
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.loss.TorchLossComputer import Hybrid_Loss
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class RhythmMambaTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.diff_flag = 0
        self.data_dict = {}
        self.dataset = config.TRAIN.DATA.DATASET
        if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            self.diff_flag = 1
        if config.TOOLBOX_MODE == "train_and_test":
            self.model = RhythmMamba().to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
            self.num_train_batches = len(data_loader["train"])
            self.criterion = Hybrid_Loss()
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            self.model = RhythmMamba().to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        elif config.TOOLBOX_MODE == "inference":
            # 初始化模型并加载预训练权重
            self.model = RhythmMamba().to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

            # 确保推理时加载预训练模型
            if not os.path.exists(config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
        else:
            raise ValueError("EfficientPhys trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            self.model.train()

            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].float(), batch[1].float()
                N, D, C, H, W = data.shape

                if self.config.TRAIN.AUG :
                    data,labels = self.data_augmentation(data,labels,batch[2],batch[3])

                data = data.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                pred_ppg = (pred_ppg-torch.mean(pred_ppg, axis=-1).view(-1, 1))/torch.std(pred_ppg, axis=-1).view(-1, 1)    # normalize

                loss = 0.0
                for ib in range(N):
                    loss = loss + self.criterion(pred_ppg[ib], labels[ib], epoch , self.config.TRAIN.DATA.FS , self.diff_flag)
                loss = loss / N
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                tbar.set_postfix(loss=loss.item())
            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))  

    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")
        print('')
        print("===Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid, labels_valid = valid_batch[0].to(self.device), valid_batch[1].to(self.device)
                N, D, C, H, W = data_valid.shape
                pred_ppg_valid = self.model(data_valid)
                pred_ppg_valid = (pred_ppg_valid-torch.mean(pred_ppg_valid, axis=-1).view(-1, 1))/torch.std(pred_ppg_valid, axis=-1).view(-1, 1)    # normalize

                for ib in range(N):
                    loss = self.criterion(pred_ppg_valid[ib], labels_valid[ib], self.config.TRAIN.EPOCHS , self.config.VALID.DATA.FS , self.diff_flag)
                    valid_loss.append(loss.item())
                    valid_step += 1
                    vbar.set_postfix(loss=loss.item())
        return np.mean(np.asarray(valid_loss))

    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===Testing===")
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        with torch.no_grad():
            predictions = dict()
            labels = dict()
            print(f"Test data size: {len(data_loader['test'])}")
            for _, test_batch in enumerate(data_loader['test']):
                batch_size = test_batch[0].shape[0]
                chunk_len = self.chunk_len
                data_test, labels_test = test_batch[0].to(self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                pred_ppg_test = self.model(data_test)
                pred_ppg_test = (pred_ppg_test - torch.mean(pred_ppg_test, axis=-1).view(-1, 1)) / torch.std(
                    pred_ppg_test, axis=-1).view(-1, 1)  # normalize
                labels_test = labels_test.view(-1, 1)
                pred_ppg_test = pred_ppg_test.view(-1, 1)
                for ib in range(batch_size):
                    subj_index = test_batch[2][ib]
                    sort_index = int(test_batch[3][ib])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[ib * chunk_len:(ib + 1) * chunk_len]
                    labels[subj_index][sort_index] = labels_test[ib * chunk_len:(ib + 1) * chunk_len]
            print(' ')
            calculate_metrics(predictions, labels, self.config)

        # 保存预测数据到文件
        output_dir = "./videoPredictions"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 添加时间戳
        np.save(os.path.join(output_dir, f"predictions_{timestamp}.npy"), predictions)
        np.save(os.path.join(output_dir, f"labels_{timestamp}.npy"), labels)
        print(f"Predictions saved to {os.path.join(output_dir, f'predictions_{timestamp}.npy')}")
        print(f"Labels saved to {os.path.join(output_dir, f'labels_{timestamp}.npy')}")

    def inference(self, data_loader):
        """Model inference on the input data (no evaluation or metrics calculation)."""
        if data_loader["inference"] is None:
            raise ValueError("No data for inference")

        print(' ')
        print("===Inference===")
        print(f"Data loader contains {len(data_loader['inference'])} slices in the test set.")
        if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
            raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")

        # 加载预训练模型
        self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
        print("Using pretrained model for inference!")
        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()

        predictions = []
        with torch.no_grad():
            print(f"Inference data size: {len(data_loader['inference'])}")
            for batch_idx, data_test in enumerate(data_loader['inference']):
                for ib in range(data_test.shape[0]):
                    # data_test 是一个张量，形状为 (1, D, C, H, W)
                    data_batch = data_test[ib]
                    # data_test = data_test.to(self.config.DEVICE)  # 将切片数据移到设备上
                    data_batch = data_batch.to(self.config.DEVICE)  # 将切片数据移到设备上

                    print("data_test shape ",data_test.shape)
                    print("data_batch shape ", data_batch.shape)
                    # 模型推理
                    pred_ppg_test = self.model(data_batch)
                    print("Model output shape:", pred_ppg_test.shape)
                    print("Model output values:", pred_ppg_test)

                    # 将预测结果转移到 CPU 并转换为 numpy 数组
                    slice_pred = pred_ppg_test.squeeze(0).detach().cpu().numpy()  # 移除批次维度

                    # 存储预测结果
                    predictions.append(slice_pred)

                    predictions_array = np.array(predictions)
                    # 重塑数组为 (n, 1) 形状，其中 n 是视频的总帧数
                    n = predictions_array.shape[0] * predictions_array.shape[1]
                    predictions_array = predictions_array.reshape(n, 1)

                    print(f"Processed batch(4 slices in 1 batch) {batch_idx}")

            print("All slices processed.")

        # 保存预测数据到文件
        output_dir = self.config.TEST.OUTPUT_SAVE_DIR
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 添加时间戳
        np.save(os.path.join(output_dir, f"predictions_{timestamp}.npy"), predictions)
        print(f"Predictions saved to {os.path.join(output_dir, f'predictions_{timestamp}.npy')}")

        # 转换为numpy数组并保存CSV
        predictions_array = np.array(predictions)
        n = predictions_array.shape[0] * predictions_array.shape[1]
        predictions_array = predictions_array.reshape(n, 1)

        csv_path = os.path.join(output_dir, f"predictions_{timestamp}.csv")
        np.savetxt(
            csv_path,
            predictions_array,
            delimiter=",",
            header="PPG_Signal",
            comments='',
            fmt='%.6f'
        )
        print(f"CSV saved to {csv_path}")

        # ________________心率统计部分___________________
        fps = 30
        # 提取第一列作为信号
        # predictions_array = np.array(predictions)
        signal = predictions_array[:, 0]

        try:
            # 动态调整峰值检测参数
            min_height = np.percentile(signal, 85)  # 使用85%分位数作为基准
            peaks, _ = find_peaks(signal, height=min_height, distance=int(fps * 0.4))  # 至少间隔0.4秒

            if len(peaks) < 2:
                # 尝试更宽松的参数
                peaks, _ = find_peaks(signal, height=min_height * 0.8, distance=int(fps * 0.3))

            if len(peaks) < 2:
                # 使用备用算法计算BPM
                fft = np.fft.rfft(signal)
                freqs = np.fft.rfftfreq(len(signal), 1 / fps)
                bpm = freqs[np.argmax(np.abs(fft))] * 60
                print(f"[备用方案] FFT计算BPM: {bpm:.1f}")
            else:
                intervals = np.diff(peaks) / fps
                bpm = 60 / np.mean(intervals)
        except Exception as e:
            print(f"心率计算失败: {str(e)}")
            bpm = 0  # 标记无效结果

        # 生成结果文件
        output_dir = self.config.TEST.OUTPUT_SAVE_DIR
        os.makedirs(output_dir, exist_ok=True)

        result_data = {
            "bmp": float(bpm),
            "image_path": os.path.abspath(os.path.join(output_dir, f"bpm_plot_{timestamp}.png")),
            "csv_path": os.path.abspath(os.path.join(output_dir, f"predictions_{timestamp}.csv")),
            "status": "completed",
            "timestamp": timestamp  # 添加时间戳作为唯一标识
        }

        # 保存图片
        plt.figure(figsize=(12, 6))
        plt.plot(signal, label="PPG Signal", color='blue')
        if len(peaks) > 0:
            plt.scatter(peaks, signal[peaks], color='red', label="Detected Peaks")
        plt.title(f"BPM Analysis Result: {bpm:.1f} BPM")
        plt.xlabel("Frame Index")
        plt.ylabel("Signal Value")
        plt.legend()
        plt.savefig(result_data["image_path"], dpi=300)
        plt.close()

        # 保存结果文件
        with open(os.path.join(output_dir, "result.json"), 'w') as f:
            json.dump(result_data, f)

        return bpm

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)

    def data_augmentation(self,data,labels,index1,index2):
        N, D, C, H, W = data.shape
        data_aug = np.zeros((N, D, C, H, W))
        labels_aug = np.zeros((N, D))
        rand1_vals = np.random.random(N)
        rand2_vals = np.random.random(N)
        for idx in range(N):
            index = index1[idx] + index2[idx]
            rand1 = rand1_vals[idx]
            rand2 = rand2_vals[idx]
            if rand1 < 0.5 :
                if index in self.data_dict:
                    gt_hr_fft = self.data_dict[index]
                else:
                    gt_hr_fft, _  = calculate_hr(labels[idx], labels[idx] , diff_flag = self.diff_flag , fs=self.config.VALID.DATA.FS)
                    self.data_dict[index] = gt_hr_fft
                    
                if gt_hr_fft > 90: 
                    rand3 = random.randint(0, D//2-1)
                    even_indices = torch.arange(0, D, 2)
                    odd_indices = even_indices + 1
                    data_aug[:, even_indices, :, :, :] = data[:, rand3 + even_indices// 2, :, :, :]
                    labels_aug[:, even_indices] = labels[:, rand3 + even_indices // 2]
                    data_aug[:, odd_indices, :, :, :] = (data[:, rand3 + odd_indices // 2, :, :, :] + data[:, rand3 + (odd_indices // 2) + 1, :, :, :]) / 2
                    labels_aug[:, odd_indices] = (labels[:, rand3 + odd_indices // 2] + labels[:, rand3 + (odd_indices // 2) + 1]) / 2
                elif gt_hr_fft < 75 :
                    data_aug[:, :D//2, :, :, :] = data[:, ::2, :, :, :]
                    labels_aug[:, :D//2] = labels[:, ::2]
                    data_aug[:, D//2:, :, :, :] = data_aug[:, :D//2, :, :, :]
                    labels_aug[:, D//2:] = labels_aug[:, :D//2]
                else :
                    data_aug[idx] = data[idx]
                    labels_aug[idx] = labels[idx]                                      
            else :
                data_aug[idx] = data[idx]
                labels_aug[idx] = labels[idx]
        data_aug = torch.tensor(data_aug).float()
        labels_aug = torch.tensor(labels_aug).float()
        if rand2 < 0.5:
            data_aug = torch.flip(data_aug, dims=[4])
        return data_aug, labels_aug