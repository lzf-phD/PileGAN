import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch

opt = TestOptions().parse(save=False)
# opt.nThreads = 1   # test code only supports nThreads = 1
opt.nThreads = 0  # test code only supports nThreads = 0
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip


opt.dataroot = 'datasets/piles_EB' #testing dataset root
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
opt.data_type = data_loader.dataset.type
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name,'%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx

for i, data in enumerate(dataset):

    if i >= opt.how_many:
        break

    minibatch = 1
    if opt.engine:
        generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
    elif opt.onnx:
        generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
    else:  # 生成器的评估
        generated, _ = model.inference(data['label'], data['label_txt'],
                                       data['image'])  # real_image为空,generated为fake_image

    visuals = OrderedDict([('input_label', util.tensor2im(data['label'][0])),
                           ('synthesized_image', util.tensor2im(generated.data[0])),
                           ('real_image', util.tensor2im(data['image'][0]))])

    img_path = data['path']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path,opt.data_type)

webpage.save()

