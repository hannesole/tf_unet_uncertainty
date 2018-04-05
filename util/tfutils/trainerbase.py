import os
import signal
import tensorflow as tf

class TrainerBase:
    """
    Class providing basic functions for training
    """
    TRAIN_LOGDIR = 'trainlogs'
    CHECKPOINTS_DIR = 'checkpoints'
    RECOVERY_CHECKPOINTS_DIR = 'recovery_checkpoints'
    CHECKPOINTS_FILE_PREFIX = 'snapshot'
    PROCESSID_FILE = 'processid'
    
    STATUS_TRAINING_FINISHED = 0
    STATUS_TRAINING_UNFINISHED = 100
    STATUS_TRAINING_NAN_LOSS = 1


    def __init__(self, session, train_dir):
        """
        session: tf.Session
            The tensorflow session for training

        train_dir: str
            Directory used for storing logs and checkpoints during training.
        """
        self._session = session
        self._coordinator = tf.train.Coordinator()
        self._summary_writer = None # move to mainloop?
        with tf.variable_scope('', reuse=False):
            self._global_step = tf.get_variable('global_step', 
                    initializer=0,
                    dtype=tf.int32,
                    trainable=False,
                    collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES],
                    )
        # self._global_step =  tf.Variable( 0, trainable=False, name="global_step", 
                # collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES] )

        # paths
        self._train_dir = train_dir
        self._train_logdir = os.path.join(self._train_dir, self.TRAIN_LOGDIR)
        self._checkpoints_dir = os.path.join(self._train_dir, self.CHECKPOINTS_DIR)
        self._recovery_checkpoints_dir = os.path.join(self._train_dir, self.RECOVERY_CHECKPOINTS_DIR)
        self._checkpoints_file_prefix = self.CHECKPOINTS_FILE_PREFIX
        self._checkpoints_path = os.path.join(self._checkpoints_dir, self._checkpoints_file_prefix)
        self._recovery_checkpoints_path = os.path.join(self._recovery_checkpoints_dir, self._checkpoints_file_prefix)

        # create dirs for logging and checkpoints
        os.makedirs(self._checkpoints_dir, exist_ok=True)
        os.makedirs(self._recovery_checkpoints_dir, exist_ok=True)
        os.makedirs(self._train_logdir, exist_ok=True)

        # setup the signal handler
        def signal_handler(signum, frame):
            print("received signal {0}".format(signum), flush=True)
            self._coordinator.request_stop()
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGUSR1, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # write pid file
        with open(os.path.join(train_dir, self.PROCESSID_FILE), 'w') as f:
            f.write(str(os.getpid()))


    #def get_checkpoints_with_time(self):
        #"""Returns a list of checkpoint files with timestamps that can passed
        #to the tf.Saver instance
        #"""
        #result = []

        #checkpoint_state = tf.train.get_checkpoint_state(self._checkpoints_dir)
        #if checkpoint_state:
            #for checkpoint_file in checkpoint_state.all_model_checkpoint_paths:
                #if os.path.exists(checkpoint_file):
                    #timestamp = os.path.getmtime(checkpoint_file)
                    #result.append((checkpoint_file,timestamp))

        #return result


    def session(self):
        """Returns the session for the trainer"""
        return self._session

    def coordinator(self):
        """Returns the coordinator for the trainer"""
        return self._coordinator

    def global_step(self):
        """Returns the tensor for the global step variable"""
        return self._global_step


