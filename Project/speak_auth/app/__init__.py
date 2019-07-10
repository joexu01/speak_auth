from flask import Flask
from flask_bootstrap import Bootstrap
from flask_mail import Mail
from flask_moment import Moment
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_uploads import UploadSet, configure_uploads, AUDIO
from config import config


# 实例化所有组件
bootstrap = Bootstrap()
mail = Mail()
moment = Moment()
db = SQLAlchemy()
# 上传文件集
user_img = UploadSet('avatar')
audio = UploadSet('audio', extensions=AUDIO)
dtw_person = UploadSet('person', extensions=AUDIO)
# 关于上传文件的设置，参见
# https://stackoverflow.com/questions/23650544/runtimeerror-cannot-access-configuration-outside-request

login_manager = LoginManager()
login_manager.login_view = 'auth.login'  # 给定登录视图


def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    # 将所有组件与应用注册
    bootstrap.init_app(app)
    mail.init_app(app)
    moment.init_app(app)
    db.init_app(app)
    login_manager.init_app(app)
    configure_uploads(app, user_img)
    configure_uploads(app, audio)
    configure_uploads(app, dtw_person)

    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    from .api import api as api_blueprint
    app.register_blueprint(api_blueprint, url_prefix='/api/v1')

    return app


def add_url_rule():
    return None
