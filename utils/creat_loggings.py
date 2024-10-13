import os, shutil, logging

def creat_loggings(experiment_dir = './experiment/3DTeethSeg/',
                   experiment_name = 'THISNet'):
    """--------------------------- create Folder ----------------------------------"""
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    file_dir = experiment_dir + experiment_name
    if os.path.exists(file_dir):
        shutil.rmtree(file_dir)
    os.makedirs(file_dir)

    log_dir, checkpoints = os.path.join(file_dir, 'logs'), os.path.join(file_dir, 'checkpoints')
    os.makedirs(log_dir)
    os.makedirs(checkpoints)

    """-------------------------- save codes --------------------------------------"""
    shutil.copytree('.', os.path.join(file_dir, 'code'),
                    ignore=shutil.ignore_patterns("experiments","experiments_cv",'dataset', '__pycache__'))

    formatter = logging.Formatter('%(name)s - %(message)s')
    logger = logging.getLogger("all")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(str(log_dir) + '/log.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return  logger, log_dir, checkpoints