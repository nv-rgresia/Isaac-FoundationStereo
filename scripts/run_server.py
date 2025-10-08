# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
import socket
import struct
import os
from PIL import Image
import io

from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *


class SAM6DClient:
  def __init__(self, host='localhost', port=8000):
    self.host = host
    self.port = port
    self.socket = None

  def connect(self):
    """Connect to the SAM-6D server"""
    try:
      self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      self.socket.connect((self.host, self.port))
      logging.info(f"Connected to SAM-6D server at {self.host}:{self.port}")
      return True
    except Exception as e:
      logging.error(f"Failed to connect to server: {str(e)}")
      return False

  def send_data(self, data):
    """Send data to the server"""
    serialized = pickle.dumps(data)
    length = struct.pack('!I', len(serialized))
    self.socket.sendall(length + serialized)

  def receive_data(self):
    """Receive data from the server"""
    # First, receive the length of the message
    length_data = b''
    while len(length_data) < 4:
      chunk = self.socket.recv(4 - len(length_data))
      if not chunk:
        raise ConnectionError("Connection closed")
      length_data += chunk

    length = struct.unpack('!I', length_data)[0]

    # Now receive the actual data
    data = b''
    while len(data) < length:
      chunk = self.socket.recv(length - len(data))
      if not chunk:
        raise ConnectionError("Connection closed")
      data += chunk

    return pickle.loads(data)

  def disconnect(self):
    """Disconnect from the server"""
    if self.socket:
      self.socket.close()

def receive_image(connection, address):
  print(f"Connection from {address} has been established!")

  image_size_data = connection.recv(4)
  image_size = struct.unpack('!I', image_size_data)[0]
  image0_data = b''
  while True:
    data = connection.recv(1048576)
    image0_data += data
    if len(image0_data) >= image_size:
      break
  img0 = cv2.imdecode(np.frombuffer(image0_data, dtype=np.uint8), cv2.IMREAD_COLOR)

  image_size_data = connection.recv(4)
  image_size = struct.unpack('!I', image_size_data)[0]
  image1_data = b''
  while True:
    data = connection.recv(1048576)
    image1_data += data
    if len(image1_data) >= image_size:
      break
  img1 = cv2.imdecode(np.frombuffer(image1_data, dtype=np.uint8), cv2.IMREAD_COLOR)

  return img0, img1

def run_inference(img0, img1, baseline, K, model, args):
  img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
  img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
  H, W = img0.shape[:2]

  img0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
  img1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
  padder = InputPadder(img0.shape, divis_by=32, force_square=False)
  img0, img1 = padder.pad(img0, img1)

  with torch.cuda.amp.autocast(True):
    if not args.hiera:
      disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
    else:
      disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
  disp = padder.unpad(disp.float())
  disp = disp.data.cpu().numpy().reshape(H, W)

  if args.remove_invisible:
    yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
    us_right = xx - disp
    invalid = us_right < 0
    disp[invalid] = np.inf

  K[:2] *= scale
  depth = K[0, 0] * baseline / disp
  depth_mm = (depth * 1000.0).astype(np.uint16)
  return Image.fromarray(depth_mm)

def start_server(args, model, host='0.0.0.0', port=12345):
  server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  server_socket.bind((host, port))
  server_socket.listen(5)
  print(f"Server listening on {host}:{port}")

  with open(args.intrinsic_file, 'r') as f:
    lines = f.readlines()
    K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3, 3)
    baseline = float(lines[1])

  sam6d_client = SAM6DClient()
  sam6d_client.connect()
  connection, address = server_socket.accept()
  while True:
    img0, img1 = receive_images(connection, address)
    depth_img = run_inference(img0, img1, baseline, K, model, args)

    # create request compress
    request = {
      'action': 'sam6d_inference',
      'rgb_bytes': img0,
      'depth_bytes': depth_img,
      'det_score_thresh': 0.5,
      'visualize': False
    }
    sam6d_client.send_data(request)
    response = sam6d_client.receive_data()

    # server_socket send size
    server_socket.send(struct.pack('!I', len(response)))
    server_socket.sendall(response)

if __name__=="__main__":
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K.txt', type=str, help='camera intrinsic matrix and baseline file')
  parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/11-33-40/model_best_bp2.pth', type=str, help='pretrained model path')
  parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
  parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
  parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
  parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
  parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
  parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
  parser.add_argument('--denoise_cloud', type=int, default=1, help='whether to denoise the point cloud')
  parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
  parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)
  os.makedirs(args.out_dir, exist_ok=True)

  ckpt_dir = args.ckpt_dir
  cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
  if 'vit_size' not in cfg:
    cfg['vit_size'] = 'vitl'
  for k in args.__dict__:
    cfg[k] = args.__dict__[k]
  args = OmegaConf.create(cfg)
  logging.info(f"args:\n{args}")
  logging.info(f"Using pretrained model from {ckpt_dir}")

  model = FoundationStereo(args)

  ckpt = torch.load(ckpt_dir, weights_only=False, map_location='cpu')
  logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
  model.load_state_dict(ckpt['model'])

  model.cuda()
  model.eval()

  scale = args.scale
  assert scale<=1, "scale must be <=1"

  start_server(args, model)
