import os

from flask_uploads import IMAGES

basedir = os.path.abspath(os.path.dirname(__file__))


# 正式上线时使用环境配置，从本地获取  e.g. CONFIG = os.environ.get('CONFIG_NAME')
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard to guess string'  # 应用秘钥
    MAIL_SERVER = 'smtp.163.com'  # 邮件服务器配置
    MAIL_PORT = int(465)  # 邮件端口
    MAIL_USERNAME = 'tyxiaoxu'
    MAIL_PASSWORD = 'flasktest01'
    MAIL_USE_TLS = False
    MAIL_USE_SSL = True
    SPEAK_AUTH_MAIL_SUBJECT_PREFIX = '[Speak Auth]'  # 邮件标题前缀
    SPEAK_AUTH_MAIL_SENDER = 'Admin <tyxiaoxu@163.com>'  # 发送者名称
    SPEAK_AUTH_ADMIN = 'tyxiaoxu@163.com'  # FLASK 管理者邮箱
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:atk_2018@localhost:3306/speak_auth'  # 数据库URI
    PRESERVE_CONTEXT_ON_EXCEPTION = False
    UPLOADED_AVATAR_DEST = 'D:/Web/speak_authentic/app/static/avatar'  # 上传头像集  最好使用绝对路径  易于维护
    UPLOADED_AUDIO_DEST = 'D:/Web/speak_authentic/app/static/temp_audio_file'
    UPLOADED_PERSON_DEST = 'D:/Web/speak_authentic/app/static/dtw_person'
    SVM_DATA_MATRIX = 'D:/Web/speak_authentic/app/static/dtw_person/data_matrix.csv'
    SVM_LABEL_MATRIX = 'D:/Web/speak_authentic/app/static/dtw_person/label_matrix.csv'
    GMM_COMMON_DATA = 'D:/Web/speak_auth/app/static/dtw_person/common_data.csv'
    GMM_THROAT_DATA = 'D:/Web/speak_auth/app/static/dtw_person/throat_data.csv'
    # 关于这个的设置，参见
    # https://stackoverflow.com/questions/23650544/runtimeerror-cannot-access-configuration-outside-request
    IMG_UPLOAD_ALLOWED = IMAGES
    SPEAK_AUTH_USER_PER_PAGE = 25
    WHOOSH_BASE = os.path.join(basedir, 'WHOOSH_BASE_INDEX')

    @staticmethod
    def init_app(app):
        pass


# 开发环境数据库配置
class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:atk_2018@localhost:3306/speak_auth'


# 测试环境数据库配置
class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:atk_2018@localhost:3306/speak_auth'


# 运营环境数据库配置
class ProductionConfig(Config):
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
                              'sqlite:///' + os.path.join(basedir, 'data.sqlite')


config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,

    'default': DevelopmentConfig
}
