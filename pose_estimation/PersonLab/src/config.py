
class config:

    #########
    # POSE CONFIGS:
    #########

    # Number of keypoints
    NUM_KP = 17

    # List of keypoint names
    KEYPOINTS = [
        "nose",         # 0
        # "neck",       
        "Rshoulder",    # 1
        "Relbow",       # 2
        "Rwrist",       # 3
        "Lshoulder",    # 4
        "Lelbow",       # 5
        "Lwrist",       # 6
        "Rhip",         # 7
        "Rknee",        # 8
        "Rankle",       # 9
        "Lhip",         # 10
        "Lknee",        # 11
        "Lankle",       # 12
        "Reye",         # 13
        "Leye",         # 14
        "Rear",         # 15
        "Lear"          # 16
    ]

    # Indices of right and left keypoints (for flipping in augmentation)
    RIGHT_KP = [1, 2, 3,  7,  8,  9, 13, 15]
    LEFT_KP =  [4, 5, 6, 10, 11, 12, 14, 16]

    # List of edges as tuples of indices into the KEYPOINTS array
    # (Each edge will be used twice in the mid-range offsets; once in each direction)
    EDGES = [
        (0, 14),
        (0, 13),
        (0, 4),
        (0, 1),
        (14, 16),
        (13, 15),
        (4, 10),
        (1, 7),
        (10, 11),
        (7, 8),
        (11, 12),
        (8, 9),
        (4, 5),
        (1, 2),
        (5, 6),
        (2, 3)
    ]

    NUM_EDGES = len(EDGES)  # 16
    
    # For better visualization, we define new connection among joints.
    # Compare with the original EDGES, we add two new CONNECTION at the end.
    NEW_CONNECTION = [(0, 14), (0, 13), (0, 4), (0, 1), (14, 16), (13, 15),
                    (4, 10), (1, 7), (10, 11), (7, 8), (11, 12), (8, 9),
                    (4, 5), (1, 2), (5, 6), (2, 3), (1, 4), (7, 10) ]
    
    NEW_NUM_EDGES = len(NEW_CONNECTION)  # 16+2

    #########
    # PRE- and POST-PROCESSING CONFIGS:
    #########

    # Radius of the discs around the keypoints. Used for computing the ground truth
    # and computing the losses. (Recommended to be a multiple of the output stride.)
    KP_RADIUS = 32

    # The threshold for extracting keypoints from hough maps.
    PEAK_THRESH = 0.004  # 0.01 in paper

    # Pixel distance threshold for whether to begin a new skeleton instance
    # (If another skeleton already has this keypoint within the threshold, it is discarded.)
    NMS_THRESH = 32  # 10 in paper

    # The metric threshold for assigning a pixel to a given instance mask 
    INSTANCE_SEG_THRESH = 0.25  # removed
    
    # Pose with joints below number_threshold will not be acknowledged 
    JOINTS_NUM_THRE = 4

    #########
    # TRAINING CONFIGS:
    #########

    # Input shape for training images (By convention s*n+1 for some integer n and s=output_stride)
    IMAGE_SHAPE = (401, 401, 3)  # here set (401, 401, 3), paper is (801, 801, 3)

    # Output stride of the base network (resnet101 or resnet152 in the paper)
    # [Any convolutional stride in the original network which would reduce the 
    # output stride further is replaced with a corresponding dilation rate.]
    OUTPUT_STRIDE = 16

    # Weights for the losses applied to the keypoint maps ('heatmap'), the binary segmentation map ('seg'),
    # and the short-, mid-, and long-range offsets.
    LOSS_WEIGHTS = {
        'heatmap': 4,
        'seg': 2,  # only for instance segmentation
        'short': 1,
        'mid': 0.25,  
        'long': 0.125  # only for instance segmentation
    }

    # The filepath for the training dataset
    H5_DATASET = '/data/zhouhuayi/data/COCO/personlab_h5py/coco2017_personlab_train.h5'
    # H5_DATASET = '/data/zhouhuayi/data/COCO/personlab_h5py/coco2017_personlab_val.h5'
    
    TRAIN_IMAGE_NUM = 64115  # COCO2017_tarin
    
    VAL_IMAGE_NUM = 2693  # COCO2017_val

    # Whether to keep the batchnorm weights frozen.
    BATCH_NORM_FROZEN = True

    # Number of GPUs to distribute across
    NUM_GPUS = 2 # 4

    # The total batch size will be (NUM_GPUS * BATCH_SIZE_PER_GPU)
    BATCH_SIZE_PER_GPU = 3

    # Whether to use Polyak weight averaging as mentioned in the paper
    POLYAK = False

    # Optional model weights filepath to use as initialization for the weights
    LOAD_MODEL_PATH = './models/personlab_model_only_pose.h5'  # './personlab_model_only_pose.h5' or None
    
    # Where to save the model.
    SAVE_MODEL_PATH = './models/personlab_model_only_pose.h5'

    # Epochs
    NUM_EPOCHS = 100
    
    # Whether use Intermediate Supervision or not
    # image input 640x640x3 with intermediate_supervision is hard to train
    INTER_SUPERVISION = False  # True or False    
    
    
    #########
    # MODEL MODE
    #########
    
    # MODE = 0: only pose estimation
    # MODE = 1: pose estimation && instance segmentation
    MODE = 0
    
    
    #########
    # TEST CONFIGS:
    #########
    
    # different mode load models, correspondingly
    if MODE is 0:
        TRAIN_MODEL_PATH = './models/personlab_model_only_pose_093.h5'
    if MODE is 1:
        TRAIN_MODEL_PATH = './models/personlab_val_125_0919.h5'
        
    TEST_IMAGES = ['./test_imgs/000000013291.jpg',
                    './test_imgs/000000000839.jpg',
                    './test_imgs/05_0014_Student.jpg',
                    './test_imgs/022_ch40_2655.jpg'
                  ]
    
    TEST_SCALE = 1.0
    

class TransformationParams:

    target_dist = 0.8
    scale_prob = 1.
    scale_min = 0.8
    scale_max = 2.0
    max_rotate_degree = 25.
    center_perterb_max = 0.
    flip_prob = 0.5
