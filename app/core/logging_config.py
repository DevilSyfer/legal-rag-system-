import logging
import os
from logging.handlers import TimedRotatingFileHandler

LOG_DIR = "logs"

def setup_logging():
    
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    file_handler = TimedRotatingFileHandler(
    filename="logs/app.log",
    when='midnight',
    interval=1,
    backupCount=7,
    encoding='utf-8'
    )
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    
    logger = logging.getLogger()
    
    logger.setLevel(logging.INFO)    
    # logger.setLevel(logging.DEBUG)  
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
            