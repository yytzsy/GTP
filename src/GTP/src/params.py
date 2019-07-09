class Params():


    gpu_id = 0

    n_epochs = 50
    iter_test = 1

    alpha_regular = 0.001
    alpha_choose = 1.0

    words_path = '../words/'
    wordtoix_path = '../words/wordtoix.npy'
    ixtoword_path = '../words/ixtoword.npy'
    word_fts_path = '../words/word_glove_fts_init.npy'
    pre_model_save_dir = '../model/'
    pre_result_save_dir = '../result/'
    word_embedding_path = '../../../dataset/glove.840B.300d_dict.npy'
    video_data_path_train = '../../../data/train/train.txt' 
    video_data_path_val = '../../../data/test/test.txt'

    dim_image = 500
    dim_word = 300
    batch_size = 20
    attn_size = 256 # RNN cell and attention module size

    # Architecture
    SRU = True # Use SRU cell, if False, use standard GRU cell
    max_video_len = 210 # Maximum number of segments in each video context
    max_sentence_len = 35 # Maximum number of words in each sentence context
    max_choose_len = 5
    num_layers = 3 # Number of layers at sentence-video matching
    num_gcn = 4
    gcn_softmax_scale = 150.0
    bias = True # Use bias term in attention

    # Training
	# NOTE: To use demo, put batch_size == 1
    mode = "train" # case-insensitive options: ["train", "test", "debug"]
    dropout = 0.2 # dropout probability, if None, don't use dropout
    zoneout = None # zoneout probability, if None, don't use zoneout
    optimizer = "adam" # Options: ["adadelta", "adam", "gradientdescent", "adagrad"]
    save_steps = 50 # Save the model at every 50 steps
    clip = True # clip gradient norm
    norm = 5.0 # global norm
    # NOTE: Change the hyperparameters of your learning algorithm here

    opt_arg = {'adam':{'learning_rate':1e-3, 'beta1':0.9, 'beta2':0.999, 'epsilon':1e-8}}
