from utils import make_gif
root_dir ='visualisation/output/'
make_gif('image', 'wgan', root_dir)
make_gif('losses', 'wgan', root_dir)

make_gif('image', 'gan', root_dir)
make_gif('losses', 'gan', root_dir)