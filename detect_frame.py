import argparse
import os
import sys
import numpy as np
import cv2
py_dll_path = os.path.join(sys.exec_prefix, 'Library', 'bin')
os.add_dll_directory(py_dll_path)
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

class Yolo():
    def __init__(self):
        self.weights = 'weights/best.pt'
        self.half = True
        self.cfg = 'cfg/yolov3-spp.cfg'
        self.imgsz = (320, 192) if ONNX_EXPORT else 512  # (320, 192) or (416, 256) or (608, 352) for (height, width)

        self.device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else '')

        self.model = Darknet(self.cfg, self.imgsz)

        # load weights
        attempt_download(self.weights)
        if self.weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['model'])
        else:  # darknet format
            load_darknet_weights(self.model, self.weights)

        self.model.to(self.device).eval()

        self.half = self.half and self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half()

        self.names = 'data/kitti_OD.names'
        self.names = list(glob.iglob('./**/' + self.names, recursive=True))[0]
        #print('***',end='')

        self.names = load_classes(self.names)
        #print(self.names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
    def forword(self, frame):
        im0 = letterbox(frame, new_shape=self.imgsz)[0]
        im0 = im0[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        im0 = np.ascontiguousarray(im0)

        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)
        _ = self.model(img.half() if self.half else img.float()) if self.device.type != 'cpu' else None  # run once
        img = torch.from_numpy(im0).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = torch_utils.time_synchronized()
        #print(img.shape)
        pred = self.model(img, augment=True)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if self.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, 0.3, 0.6,
                                   multi_label=False, agnostic=True)

        gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]
        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                #print(det[:,:4])
        try:
            result = pred[0].cpu().detach().numpy()
        except Exception as E:
            print(pred[0])
            return []
        #print(result)
        return result





def detect(save_img=False):
    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                          input_names=['images'], output_names=['classes', 'boxes'])

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    print('-------------'+str(imgsz))
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                #print(det)
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                #print(det)
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        print(xywh)
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        print(xyxy)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
#     parser.add_argument('--names', type=str, default='data/kitti_OD.names', help='*.names path')
#     parser.add_argument('--weights', type=str, default='weights/best.pt', help='weights path')
#     parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
#     parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
#     parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
#     parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
#     parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
#     parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
#     parser.add_argument('--view-img', action='store_true', help='display results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     opt = parser.parse_args()
#     opt.cfg = list(glob.iglob('./**/' + opt.cfg, recursive=True))[0]  # find file
#     opt.names = list(glob.iglob('./**/' + opt.names, recursive=True))[0]  # find file
# #     print(opt)
#
#     with torch.no_grad():
#         detect()

a = Yolo()
#frame = cv2.imread('./data/samples/000002.png')
path = 'H:/Trackingset'
filelist = os.listdir(path)
print(filelist)
file_array = []
for i in range(len(filelist)):
    npath = path+'/'+filelist[i]
    filelist2 = os.listdir(npath)
    for j in filelist2:
        file_array.append(npath+'/'+j)
#print(frame.shape)
for k in file_array:
    #if k != 'H:/Trackingset/0001/000142.png':
    #    continue
    print(k)
    frame = cv2.imread(k)
    result = a.forword(frame)
    for i in range(len(result)):
        #if i == 7:
        #    continue
        #print(result[i])
        #if result[i][5] == 2:
        #    t= 'Car'
        #elif result[i][5] == 0:
        #    t='Pedestrian'
        #else :
        #    t = 'Van'

        frame = cv2.rectangle(frame, (result[i][0],result[i][1]), (result[i][2],result[i][3]),(0,0,255),3)
        #frame = cv2.putText(frame, t, (int(result[i][0]), int(result[i][1])), 1, 2, (0, 255, 255), 2)
    cv2.imshow("asd", frame)

    #print(result)

    cv2.waitKey(1)
#if cv2.waitKey(1) == ord('q'):
#    break
