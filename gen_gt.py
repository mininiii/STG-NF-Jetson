
import json
import numpy as np
import os
import argparse

'''
<example>
python3 gen_gt.py --train_label /home/myyang/projects/dataset/AIHub/zipfiles/training/label/invation/look_inside 
--test_label /home/myyang/projects/dataset/AIHub/zipfiles/validation/unzip/look_inside/label
'''

# train 영상 데이터셋에 대한 label 파일 경로
# TRAIN_LABEL_PATH = '/home/myyang/projects/dataset/AIHub/zipfiles/training/label/invation/look_inside'
# # test(validation) 영상 데이터셋에 대한 label 파일 경로
# TEST_LABEL_PATH = '/home/myyang/projects/dataset/AIHub/zipfiles/validation/unzip/look_inside/label'

## 수정 X
# 무조건 이 경로에 있어야 함. 학습시에 사용하기 위해
# gt 저장할 폴더 경로
OUTPUT_PATH = './data/AIHub/gt/'
# alphapose로 생성된 .json파일 폴더 경로
POSE_PATH = f'./data/AIHub/pose/'

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process label paths.')
    parser.add_argument('--train_label', type=str, required=False, help='Path to the train label directory.')
    parser.add_argument('--test_label', type=str, required=False, help='Path to the test label directory.')
    return parser.parse_args()

def process_json_file(directory_path, json_file):
    file_name = json_file.replace("_alphapose_tracked_person", '')
    json_file_path = os.path.join(directory_path, file_name)
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    target = json_file.split('_')[1]
    # block_detail이 target("A20", ...)인 것만 찾기
    frames_count = int(data['file'][0]['videos']['block_information'][-1]['end_frame_index']) + 1

    ground_truth = np.zeros(frames_count)

    for block in data['file'][0]['videos']['block_information']:
        if block['block_detail'] == target:
            start_frame = int(block['start_frame_index'])
            end_frame = int(block['end_frame_index'])
            ground_truth[start_frame:end_frame + 1] = 1

    # 넘파이 파일로 저장
    output_file_path = os.path.join(OUTPUT_PATH, file_name[:-5])
    np.save(output_file_path, ground_truth)
    print(f"Processed: {json_file_path}")

def find_files(type, path):
# 디렉토리 내의 모든 .json 파일 찾기
    directory_path = f'{POSE_PATH}{type}'
    json_files = [f for f in os.listdir(directory_path) if f.endswith('alphapose_tracked_person.json')]
    
    # 학습할 pose 파일에 맞는 labal 찾기
    filepath = path
    for json_file in json_files:
        process_json_file(filepath, json_file)


args = parse_arguments()

if args.train_label:
    train_label_path = args.train_label
    find_files('train', train_label_path)
if args.test_label:
    test_label_path = args.test_label
    find_files('test', test_label_path)
