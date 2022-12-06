import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pth_file',
                        nargs='?',
                        type=str,
                        default='checkpoint.pth',
                        help='model path')
    parser.add_argument('-s',
                        '--source',
                        type=str,
                        default='data/images',
                        help='file/dir/0(webcam)')
    parser.add_argument('--gpu',
                        default=None,
                        type=int,
                        help='GPU id to use. default use cpu')
    opt = parser.parse_args()

    model: nn.Module = load_model(opt.pth_file, gpu=opt.gpu)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    source = str(opt.source)
    is_file = os.path.splitext(source)[1] in (".jpg", ".png")
    webcam = source.isnumeric() and not is_file

    if is_file:
        image = cv2.imread(source, flags=1)
        label, y = test(model, image)
        print(f"{source}: [{os.path.basename(source)} <--> {label}] ({y})")
    elif webcam:
        stream_processing(model, test)
    else:
        for i in os.listdir(source):
            if os.path.isfile(os.path.join(source, i)):
                image = cv2.imread(os.path.join(source, i), flags=1)
                label, y = test(model, image)
                print(f"{i}: [{os.path.basename(source)} <--> {label}] ({y})")


def load_model(pth_path: str, gpu=None, loc="cpu") -> nn.Module:
    print("=> loading checkpoint '{}'".format(pth_path))
    if gpu is None:
        checkpoint = torch.load(pth_path, map_location=loc)
    elif torch.cuda.is_available():
        # Map model to be loaded to specified single opt..
        loc = 'cuda:{}'.format(gpu)
        checkpoint = torch.load(pth_path, map_location=loc)

    args = checkpoint["args"]
    print("=> creating model '{}' on {}".format(args.arch, loc))
    for p in checkpoint['model'].parameters():
        p.requires_grad = False

    # load model, checkpoint state_dict as FP32
    model: nn.Module = checkpoint['model'].float()
    print(f"=> loaded checkpoint '{pth_path}' (epoch {checkpoint['epoch']})")

    return model


def stream_processing(model, test_cb):
    import time
    # fcap = cv2.VideoCapture('demo.mp4')
    fcap = cv2.VideoCapture(0)

    w = fcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = fcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = fcap.get(cv2.CAP_PROP_FPS)
    fcount = fcap.get(cv2.CAP_PROP_FRAME_COUNT)

    # 获取VideoWriter类实例
    # writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), int(fps), (int(w), int(h)))

    last_time = time.time()

    while fcap.isOpened():
        success, frame = fcap.read()
        while success:
            label, y = test_cb(model, frame)
            interval = time.time() - last_time
            last_time = time.time()
            cv2.putText(
                frame, "fps:%.2f label:%s, (%.3f)" % (1. / interval, label, y),
                (0, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1,
                cv2.LINE_4)
            cv2.imshow("demo", frame)  ## 显示画面
            # 获取帧画面
            success, frame = fcap.read()

            # 保存帧数据
            # writer.write(frame)

            if (cv2.waitKey(20) & 0xff
                ) == ord('q'):  ## 等待20ms并判断是按“q”退出，相当于帧率是50hz，注意waitKey只能传入整数，
                break
        fcap.release()
    # writer.release()
    cv2.destroyAllWindows()  ## 销毁所有opencv显示窗口


def test(model: nn.Module, image) -> list:
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image2 = cv2.resize(image1, (224, 224))
    # cv2.imshow('imshow',image)
    # cv2.imshow('imshow1',image2)
    img = transforms.ToTensor()(image2)
    img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])(img)
    img = img.unsqueeze(0)  # 增加一个维度
    # img = Variable(img)
    y_pred = model(img)
    smax = nn.Softmax(1)
    smax_out = smax(y_pred)
    y = smax_out.cpu().data.numpy()
    pred_label = np.argmax(y)
    # print("pred_label ",pred_label, y[0][pred_label])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return pred_label, y[0][pred_label]


if __name__ == "__main__":
    main()
