import os
import argparse
import dlib
from tqdm import tqdm
from os.path import join
from models.Xception import Xception
from models.MosNet import MosNet
from pipeline import *


def main(video_path, model_path, output_path,
         start_frame=0, end_frame=None, cuda=True):
    """
    read videos and evaluate a subset of frames with pretrained models
    ----------------------------------------------------------------------
    param:
        video_path: path to video file
        model_path: path to model file
        output_path: path where the output video is stored
        start_frame: first frame to evaluate
        end_frame: last frame to evaluate
        cuda: enable cuda
    return:
        None
    -----------------------------------------------------------------------
    """
    print('videos: {}'.format(video_path))

    # read and write
    reader = cv2.VideoCapture(video_path)

    # name for result
    video_res = video_path.split('/')[-1].split('.')[0]+'.avi'

    os.makedirs(output_path, exist_ok=True)

    # code used to compress the frames
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    # face detector
    face_detector = dlib.get_frontal_face_detector()

    # load model
    if model_path is not None:
        net = MosNet()
        net.load_state_dict(torch.load(model_path, map_location="cuda:0"))
        # net = torch.load(model_path, map_location="cuda:0")
        print('Model found in {}'.format(model_path))
    else:
        print('please check model_path')
    if cuda:
        net = net.cuda()

    # text variables for result visualization
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    # frame numbers and length of output video
    frame_num = 0
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame-start_frame)

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1

        if frame_num < start_frame:
            continue
        pbar.update(1)

        # image size
        height, width = image.shape[:2]

        # init writer
        if writer is None:
            writer = cv2.VideoWriter(join(output_path, video_res), fourcc, fps,
                                     (height, width)[::-1])

        # detect with dlib
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 1)
        if len(faces):
            # for now only take biggest face
            face = faces[0]

            # -------------------- prediction -------------------------------------
            # enlarge faces and check bounds
            x, y, size = get_boundingbox(face, width, height)
            cropped_face = image[y:y+size, x:x+size]

            # prediction using  model
            prediction, output = predict_with_model(cropped_face, net, cuda=cuda)
            # ----------------------------------------------------------------------

            # text and bb
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            label = 'fake' if prediction == 1 else 'real'
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
            output_list = ['{0:.2f}'.format(float(x)) for x in
                           output.detach().cpu().numpy()[0]]
            cv2.putText(image, str(output_list)+'=>'+label, (x, y+h+30),
                        font_face, font_scale,
                        color, thickness, 2)
            # draw box over face
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        if frame_num >= end_frame:
            break

        # Show
        cv2.imshow('test', image)
        cv2.waitKey(33)     # About 30 fps
        writer.write(image)
    pbar.close()
    if writer is not None:
        writer.release()
        print('Finished! Output saved under {}'.format(output_path))
    else:
        print('Input video file was empty')


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-v', type=str, default='./videos/2.mp4')
    p.add_argument('--model_path', '-m', type=str, default='./models/mosNet_dict.pkl')
    p.add_argument('--output_path', '-o', type=str, default='./results')
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--end_frame', type=int, default=None)
    p.add_argument('--cuda', default=True, action='store_true')
    args = p.parse_args()

    video_path = args.video_path
    if video_path.endswith('.mp4') or video_path.endswith('.avi'):
        main(**vars(args))
    else:
        # set for more than one videos
        videos = os.listdir(video_path)
        for video in videos:
            args.video_path = join(video_path, video)
            main(**vars(args))