import cv2
import torch
import numpy as np
from ultralytics import YOLO
from models.STG_NF.model_pose import STG_NF
from utils.train_utils import init_model_params

from torch.utils.tensorboard import SummaryWriter
from models.STG_NF.model_pose import STG_NF
from models.training import Trainer
from utils.data_utils import trans_list
from utils.optim_init import init_optimizer, init_scheduler
from args import create_exp_dirs
from args import init_parser, init_sub_args
from dataset import get_dataset_and_loader
from utils.train_utils import dump_args
from utils.scoring_utils import score_dataset
from utils.train_utils import calc_num_of_params

def init_model_params(args):
    return {
        'pose_shape': (2, 24, 18), # 추후 수정
        'hidden_channels': args.model_hidden_dim,
        'K': args.K,
        'L': args.L,
        'R': args.R,
        'actnorm_scale': 1.0,
        'flow_permutation': args.flow_permutation,
        'flow_coupling': 'affine',
        'LU_decomposed': True,
        'learn_top': False,
        'edge_importance': args.edge_importance,
        'temporal_kernel_size': args.temporal_kernel,
        'strategy': args.adj_strategy,
        'max_hops': args.max_hops,
        'device': args.device,
    }

# yolov8-pose 모델 로드
yolov8_model = YOLO('yolov8n-pose.pt')

# STG-NF 모델 초기화
args = init_parser().parse_args()  # args를 초기화하거나 직접 설정할 수 있습니다
_, model_args = init_sub_args(args)
model_args = init_model_params(args)
stg_nf_model = STG_NF(**model_args)
# .to('cuda' if torch.cuda.is_available() else 'cpu')
stg_nf_model.eval()  # 평가 모드로 전환

# 체크포인트 로드 (필요할 경우)
checkpoint_path = '/workspace/STG-NF/checkpoints/ShanghaiTech_85_9.tar' # data/exp_dir/AIHub/Sep19_1201/Sep19_1203__checkpoint.pth.tar
checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
stg_nf_model.load_state_dict(checkpoint['state_dict'])

# 카메라 스트림 시작
cap = cv2.VideoCapture(0)
pose_buffer = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 포즈 추출
    results = yolov8_model(frame)
    keypoints_frame = []

    if results.keypoints is not None:
        for i in range(len(results.keypoints.xy)):
            keypoints = results.keypoints.xy[i].cpu().numpy()  # (num_keypoints, 2)
            keypoint_scores = results.keypoints.conf[i].cpu().numpy()  # (num_keypoints,)

            keypoints_with_scores = []
            for j in range(len(keypoints)):
                keypoints_with_scores.extend([keypoints[j][0], keypoints[j][1], keypoint_scores[j]])

            keypoints_frame.append(keypoints_with_scores)

    if keypoints_frame:
        pose_buffer.append(keypoints_frame)

    # 포즈 데이터를 일정 프레임 수만큼 누적하여 STG-NF 모델에 입력
    if len(pose_buffer) == 30:  # 예를 들어 30프레임을 누적
        pose_tensor = torch.tensor(pose_buffer).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # STG-NF 모델로 예측 수행
        with torch.no_grad():
            prediction = stg_nf_model(pose_tensor)

        # 결과를 사용하거나 출력
        print("STG-NF Prediction:", prediction)
        
        # 사용 후 버퍼 초기화 또는 슬라이딩 윈도우 방식으로 갱신
        pose_buffer = []

    # 실시간 프레임 디스플레이
    cv2.imshow('Pose Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
