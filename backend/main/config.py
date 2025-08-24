# 数据库配置
DB_USER = 'root'
DB_PASSWORD = '123456'
DB_HOST = '172.29.224.1'
DB_NAME = 'showphys'

# 数据库 URI
SQLALCHEMY_DATABASE_URI = f'mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'
SQLALCHEMY_TRACK_MODIFICATIONS = False

# 上传文件配置
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'webm'}