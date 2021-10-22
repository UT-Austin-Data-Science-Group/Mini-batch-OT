import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # hardware config
    parser.add_argument(
        '--gpu_id',
        type=str,
        default='0',
        help='GPU ids to use (separated by comma e.g. 0,1,2,3)')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='number of dataloader workers')
    # training config
    parser.add_argument(
        '--method',
        type=str,
        default='jdot',
        choices=['jdot', 'jumbot', 'jpmbot'],
        help='model name')
    parser.add_argument(
        '--use_bomb', 
        action='store_true', 
        help='whether to use BomB version')
    parser.add_argument(
        '--source_ds', 
        type=str, 
        default='svhn',  
        help="The source dataset")
    parser.add_argument(
        '--target_ds', 
        type=str, 
        default='mnist',  
        help="The target dataset")
    parser.add_argument(
        '--data_dir', 
        type=str, 
        default='./data', 
        help="Data directory")
    parser.add_argument(
        '--generator',
        type=str,
        default='Cnn',
        choices=['Cnn', 'Resnet50'],
        help='generator name')
    parser.add_argument(
        '--k',
        type=int,
        default=1,
        help='number of minibatches')
    parser.add_argument(
        '--mbsize',
        type=int,
        default=500,
        help='minibatch size')
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=100,
        help='number of epoch at k=1')
    parser.add_argument(
        '--n_iterations',
        type=int,
        default=10000,
        help='number of iterations for Office-Home and VisDA')
    parser.add_argument(
        '--test_interval',
        type=int,
        default=1,
        help='interval of two continuous test phase')
    parser.add_argument(
        '--nclass',
        type=int,
        default=10,
        help='number of classes')
    parser.add_argument(
        '--bottleneck_dim',
        type=int,
        default=512,
        help='number of output features of Resnet50 generator')
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0,
        help='OT regularization coefficient')
    parser.add_argument(
        '--batch_epsilon', 
        type=float, 
        default=0., 
        help="OT regularization coefficient between minibatches")
    parser.add_argument(
        '--tau',
        type=float,
        default=1,
        help='marginal penalization coeffidient')
    parser.add_argument(
        '--mass',
        type=float,
        default=1,
        help='ratio of masses to be transported')
    parser.add_argument(
        '--lr',
        type=float,
        default=2e-4,
        help='learning rate')
    parser.add_argument(
        '--seed',
        type=int,
        default=1980,
        help='random seed')
    parser.add_argument(
        '--eta1',
        type=float,
        default=0.1,
        help='weight of embedding loss')
    parser.add_argument(
        '--eta2',
        type=float,
        default=0.1,
        help='weight of transportation loss')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='momentum for SGD optimizer')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0005,
        help='weight decay for SGD optimizer')
    parser.add_argument(
        '--lr_mult',
        type=float,
        default=10,
        help='ratio of classifier lr to generator lr')
    parser.add_argument(
        '--decay_mult',
        type=float,
        default=2,
        help='ratio of classifier weight decay to generator weight decay')
    parser.add_argument(
        '--gamma',
        type=float,
        default=10,
        help='param for inverse learning rate scheduler')
    parser.add_argument(
        '--power',
        type=float,
        default=0.75,
        help='param for inverse learning rate scheduler')
    
    args = parser.parse_args()
    
    return args