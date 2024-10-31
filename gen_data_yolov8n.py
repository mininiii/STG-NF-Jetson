import os
import json
import argparse
from ultralytics import YOLO  # yolov8_pose model
#TODO 거울 비친거, 무기 든거 동영상 결과 추출해서 비교해보기
#TODO iou, conf 바꾼거 비교해보기
def convert_data_format(data, split='None'):
    # 숫자 자릿수 설정
    if split == 'testing':
        num_digits = 3
    elif split == 'training':
        num_digits = 4
    elif split == 'None':
        num_digits = 4

    data_new = dict()
    for item in data:
        frame_idx_str = item['image_id'][:-4]  # '0.jpg' -> '0'
        frame_idx_str = frame_idx_str.zfill(num_digits)  # '0' -> '000'
        person_idx_str = str(item['idx'])
        keypoints = item['keypoints']
        scores = item['score']
        
        # 새로운 데이터 구조 생성
        if person_idx_str not in data_new:
            data_new[person_idx_str] = {frame_idx_str: {'keypoints': keypoints, 'scores': scores}}
        else:
            data_new[person_idx_str][frame_idx_str] = {'keypoints': keypoints, 'scores': scores}

    return data_new

def read_convert_write(in_full_fname, out_full_fname):
    with open(in_full_fname, 'r') as f:
        data = json.load(f)

    data_new = convert_data_format(data)

    save = True
    if save:
        with open(out_full_fname, 'w') as f:
            json.dump(data_new, f)

def run_yolov8n_pose(video_path, file_name, output_dir, is_video=False):
    model = YOLO('yolov8n-pose.pt')  # Load the pre-trained YOLOv8n pose model
    results = model.track(source=video_path, save=False, save_txt=False, save_conf=False, stream=True, iou=0.6, conf=0.1)
    # Create results in the required format
    dets_results = []
    for frame_idx, result in enumerate(results):
        # keypoints 정보가 있는지 확인
        if result.keypoints is not None and result.boxes.is_track:
            for i in range(len(result.keypoints.xy)):
                # keypoints를 [x, y, confidence] 형식으로 변환
                keypoints_with_scores = []
                keypoints = result.keypoints.xy[i].cpu().numpy()  # (num_keypoints, 2)
                keypoint_scores = result.keypoints.conf[i].cpu().numpy()  # (num_keypoints,)
                
                for j in range(len(keypoints)):
                    keypoints_with_scores.append(float(keypoints[j][0]))  # x 좌표
                    keypoints_with_scores.append(float(keypoints[j][1]))  # y 좌표
                    keypoints_with_scores.append(float(keypoint_scores[j]))  # confidence score

                det_dict = {
                    "image_id": f"{frame_idx}.jpg",  # 프레임 인덱스를 기반으로 image_id 설정
                    "category_id": 1,  # 'person' 카테고리 가정
                    "keypoints": keypoints_with_scores,  # 변환된 keypoints 리스트
                    "score": float(result.boxes.conf[i].item()) if result.boxes.conf is not None else None,  # confidence score
                    "box": result.boxes.xywh[i].cpu().numpy().tolist(), # 변환된 box 좌표
                    "idx": int(result.boxes.id[i].item())  # 객체 인덱스
                }
                dets_results.append(det_dict)
    
    # Save results as JSON
    result_json_path = os.path.join(output_dir, f'{file_name}_alphapose-results.json')
    with open(result_json_path, 'w') as f:
        json.dump(dets_results, f)
    # with open(result_json_path, 'w') as json_file:
    #     json.dump(dets_results, json_file, indent=4)

    # Convert results to the final format
    converted_json_path = os.path.join(output_dir, f'{file_name}_alphapose_tracked_person.json')
    read_convert_write(result_json_path, converted_json_path)

def main():
    # Argument parsing
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', dest='dir', type=str, required=True, help='Directory of videos/images')
    ap.add_argument('--outdir', dest='outdir', type=str, required=True, help='Output directory')
    ap.add_argument('--video', dest='video', action='store_true', help='Is input a video file?')
    args = ap.parse_args()

    root = args.dir
    out_dir = args.outdir
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # 이미 완료된 파일에 대해서는 하고싶지 않을 때 활성화
    done_files = next(os.walk(out_dir))[2]
    done_files_unique = set(done_files)
    for file_name in done_files:
        # 확장자나 트래킹/결과 파일 이름을 제거하고 mp4 확장자 추가
        base_name = file_name.split('_alphapose')[0] + ".mp4"
        done_files_unique.add(base_name)
        
    

    for path, subdirs, files in os.walk(root):
        for name in files:
            if name in done_files_unique: continue
            # if name != "C021_A19_SY32_P01_S05_01DBS.mp4": continue
            run_pose = False
            if args.video and (name.endswith(".mp4") or name.endswith(".avi")):
                video_filename = os.path.join(path, name)
                run_pose = True
            elif name.endswith(".jpg") or name.endswith(".png"):
                video_filename = os.path.join(path, name)
                run_pose = True

            if run_pose:
                print(f'Processing {video_filename}...')
                run_yolov8n_pose(video_filename, name[:-4], out_dir, is_video=args.video)

if __name__ == '__main__':
    main()
