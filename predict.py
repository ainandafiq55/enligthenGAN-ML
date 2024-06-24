import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html

opt = TestOptions().parse()
opt.nThreads = 0   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join("./ablation/", opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
print(len(dataset))
total_time = 0
for i, data in enumerate(dataset):
    start_time = time.time()
    
    model.set_input(data)
    visuals = model.predict()
    
    end_time = (time.time() - start_time)*1000
    total_time += end_time
    
    img_path = model.get_image_paths()
    print('process image... %s, %d ms' % (img_path,end_time))
    visualizer.save_images(webpage, visuals, img_path)
average_time = total_time/len(dataset)
print('All Done, average process time : %d ms' % average_time)
webpage.save()
